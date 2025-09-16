from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import blend_input_frame

from ..components.Component import SpatioTemporalLSTMCellV2


class PredRnnV2(nn.Module):
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
        super().__init__()
        self.frame_channel: int = patch_size * patch_size * img_channel
        self.num_layers: int = len(num_hidden)
        self.num_hidden: List[int] = num_hidden
        self.patch_height: int = image_height // patch_size
        self.patch_width: int = image_width // patch_size
        self.mse_loss: nn.MSELoss = nn.MSELoss()

        self.cell_list: nn.ModuleList = nn.ModuleList(
            [
                SpatioTemporalLSTMCellV2(
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

        # V2
        # shared adapter
        adapter_num_hidden: int = num_hidden[0]
        self.adapter: nn.Conv2d = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False
        )

    def forward(
        self, frames_tensor: torch.Tensor, mask_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # V2
        delta_c_list = [
            torch.zeros(batch_size, self.num_hidden[i], height, width).to(
                frames_tensor.device
            )
            for i in range(self.num_layers)
        ]
        # V2
        delta_m_list = [
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
        decouple_loss = []
        for t in range(totalLength - 1):
            cellInput = blend_input_frame(frames_tensor, mask_true, t, generated_frame)

            # V2
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](
                cellInput, h_t[0], c_t[0], memory
            )
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                dim=2,
            )
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                dim=2,
            )

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1),
                    dim=2,
                )
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1),
                    dim=2,
                )

            generated_frame: torch.Tensor = self.conv_last(h_t[-1])
            generated_frames.append(generated_frame)

            # V2
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(
                        torch.abs(
                            torch.cosine_similarity(
                                delta_c_list[i], delta_m_list[i], dim=2
                            )
                        )
                    )
                )

        targetLength = mask_true.size(1) + 1
        output_frames = (
            torch.stack(generated_frames[-targetLength:], dim=0)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        return output_frames, decouple_loss
