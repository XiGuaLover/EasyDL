from typing import List

import torch
import torch.nn as nn
from utils.tools import blend_input_frame

from ..components.Component import SpatioTemporalLSTMCell


class PredRNN(nn.Module):
    def __init__(
        self,
        num_hidden: List[int],
        patch_size: int,
        img_channel: int,
        image_height: int,
        image_width: int,
        kernel_size: int,
        stride: int,
        layer_norm: bool,
    ) -> None:
        super(PredRNN, self).__init__()
        self.frame_channel: int = patch_size * patch_size * img_channel
        self.num_layers: int = len(num_hidden)
        self.num_hidden: List[int] = num_hidden
        self.patch_height: int = image_height // patch_size
        self.patch_width: int = image_width // patch_size
        self.mse_loss: nn.MSELoss = nn.MSELoss()

        self.cell_list: nn.ModuleList = nn.ModuleList(
            [
                SpatioTemporalLSTMCell(
                    in_channel=self.frame_channel if i == 0 else num_hidden[i - 1],
                    num_hidden=num_hidden[i],
                    height=self.patch_height,
                    width=self.patch_width,
                    kernelSize=kernel_size,
                    stride=stride,
                    layer_norm=layer_norm,
                )
                for i in range(self.num_layers)
            ]
        )
        self.conv_last: nn.Conv2d = nn.Conv2d(
            num_hidden[-1],
            self.frame_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> torch.Tensor:
        batch_size, totalLength, _, height, width = frames_tensor.shape

        h_t = [
            torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                frames_tensor.device
            )
            for i in range(self.num_layers)
        ]
        c_t = [
            torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                frames_tensor.device
            )
            for i in range(self.num_layers)
        ]
        memory: torch.Tensor = torch.zeros(
            [batch_size, self.num_hidden[0], height, width]
        ).to(frames_tensor.device)

        generated_frames: List[torch.Tensor] = []
        generated_frame = None
        for t in range(totalLength - 1):
            cellInput = blend_input_frame(frames_tensor, mask_true, t, generated_frame)

            h_t[0], c_t[0], memory = self.cell_list[0](
                cellInput, h_t[0], c_t[0], memory
            )
            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
            generated_frame: torch.Tensor = self.conv_last(h_t[-1])
            generated_frames.append(generated_frame)

        targetLength = mask_true.size(1) + 1
        output_frames = (
            torch.stack(generated_frames[-targetLength:], dim=0)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

        return output_frames
