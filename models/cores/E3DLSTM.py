from typing import List, Tuple

import torch
import torch.nn as nn
from utils.tools import blend_input_frame

from ..components.Component import Eidetic3DLSTMCell


class E3DLSTM(nn.Module):
    def __init__(
        self,
        num_hidden: List[int],
        patch_size: int,
        img_channel: int,
        window_length: int,
        window_stride: int,
        image_height: int,
        image_width: int,
        kernel_size: Tuple[int, int, int],
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(E3DLSTM, self).__init__()
        self.frame_channel: int = patch_size * patch_size * img_channel
        self.num_layers: int = len(num_hidden)
        self.num_hidden: List[int] = num_hidden
        self.window_length: int = window_length
        self.window_stride: int = window_stride
        self.patch_height: int = image_height // patch_size
        self.patch_width: int = image_width // patch_size
        self.mse_loss: nn.MSELoss = nn.MSELoss()

        self.cell_list: nn.ModuleList = nn.ModuleList(
            [
                Eidetic3DLSTMCell(
                    in_channel=self.frame_channel if i == 0 else num_hidden[i - 1],
                    num_hidden=num_hidden[i],
                    window_length=window_length,
                    height=self.patch_height,
                    width=self.patch_width,
                    kernelSize=kernel_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
                for i in range(self.num_layers)
            ]
        )
        self.conv_last: nn.Conv3d = nn.Conv3d(
            num_hidden[-1],
            self.frame_channel,
            kernel_size=(window_length, 1, 1),
            stride=(window_length, 1, 1),
            padding=0,
            bias=False,
        )

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> torch.Tensor:
        batch_size, totalLength, _, height, width = frames_tensor.shape

        input_list = [
            torch.zeros_like(frames_tensor[:, 0]) for i in range(self.window_length - 1)
        ]

        h_t = [
            torch.zeros(
                batch_size, self.num_hidden[i], self.window_length, height, width
            ).to(frames_tensor.device)
            for i in range(self.num_layers)
        ]
        c_t = [
            torch.zeros(
                batch_size, self.num_hidden[i], self.window_length, height, width
            ).to(frames_tensor.device)
            for i in range(self.num_layers)
        ]
        c_history = [
            torch.zeros(
                batch_size, self.num_hidden[i], self.window_length, height, width
            ).to(frames_tensor.device)
            for i in range(self.num_layers)
        ]

        memory: torch.Tensor = torch.zeros(
            [batch_size, self.num_hidden[0], self.window_length, height, width]
        ).to(frames_tensor.device)

        generated_frames: List[torch.Tensor] = []
        generated_frame = None
        for t in range(totalLength - 1):
            cellInput = blend_input_frame(frames_tensor, mask_true, t, generated_frame)
            input_list.append(cellInput)

            if t % (self.window_length - self.window_stride) == 0:
                cellInput = torch.stack(input_list[t:], dim=0)
                # that shape will be [depth, batch, patchChannel, patchHeight, patchWidth] -> [batch,  patchChannel, depth, patchHeight, patchWidth]
                cellInput = cellInput.permute(1, 2, 0, 3, 4).contiguous()

            for i in range(self.num_layers):
                c_history[i] = (
                    c_t[i] if i == 0 else torch.cat((c_history[i], c_t[i]), dim=1)
                )
                input = cellInput if i == 0 else h_t[i - 1]
                h_t[i], c_t[i], memory = self.cell_list[i](
                    input, h_t[i], c_t[i], memory, c_history[i]
                )
            # the shape is like this [batch, patchChannel, 1, patchHeight, patchWidth], so need to remove dim 2
            generated_frame: torch.Tensor = self.conv_last(h_t[-1]).squeeze(2)
            generated_frames.append(generated_frame)

        targetLength = mask_true.size(1) + 1
        output_frames = (
            torch.stack(generated_frames[-targetLength:], dim=0)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

        return output_frames
