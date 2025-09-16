from typing import List, Tuple

import torch
import torch.nn as nn
from utils.tools import blend_input_frame

from ..components.Component import ConvLSTMCell


class ConvLSTM(nn.Module):
    def __init__(
        self,
        num_hidden: List[int],
        kernel_size: Tuple[int, int],
        img_channel: int,
        patch_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.hidden_dim: List[int] = num_hidden
        self.num_layers: int = len(num_hidden)
        frame_channel: int = patch_size * patch_size * img_channel

        # Initialize ConvLSTM cells for each layer
        cell_list: List[ConvLSTMCell] = []
        for i in range(self.num_layers):
            cur_input_dim = frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    inputFeatureDimensions=cur_input_dim,
                    hiddenFeatureDimensions=num_hidden[i],
                    stride=stride,
                    kernelSize=kernel_size,
                    bias=False,
                )
            )
        self.cell_list: nn.ModuleList = nn.ModuleList(cell_list)

        self.final_conv = nn.Conv2d(
            in_channels=num_hidden[-1],
            out_channels=frame_channel,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

    def forward(
        self,
        frames_tensor: torch.Tensor,
        mask_true: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, total_length, _, height, width = frames_tensor.shape

        h_t = [
            torch.zeros(batch_size, self.hidden_dim[i], height, width).to(
                frames_tensor.device
            )
            for i in range(self.num_layers)
        ]
        c_t = [
            torch.zeros(batch_size, self.hidden_dim[i], height, width).to(
                frames_tensor.device
            )
            for i in range(self.num_layers)
        ]

        generated_frames: List[torch.Tensor] = []
        generated_frame = None
        for t in range(total_length - 1):
            cellInput = blend_input_frame(frames_tensor, mask_true, t, generated_frame)

            h_t[0], c_t[0] = self.cell_list[0](
                inputTensor=cellInput, curState=(h_t[0], c_t[0])
            )
            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](
                    inputTensor=h_t[i - 1], curState=(h_t[i], c_t[i])
                )
            generated_frame: torch.Tensor = self.final_conv(h_t[-1])
            generated_frames.append(generated_frame)

        targetLength = mask_true.size(1) + 1
        output = (
            torch.stack(generated_frames[-targetLength:], 0)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

        return output
