from typing import List, Tuple

import torch
import torch.nn as nn
from utils.tools import blend_input_frame

from ..components.PhyDComponent import K2M, PhyCell, PhyD_ConvLSTM, PhyD_EncoderRNN


class PhyDNet(nn.Module):
    def __init__(
        self,
        phy_cell_num_hidden: List[int],
        conv_num_hidden: List[int],
        patch_size: int,
        img_channel: int,
        image_height: int,
        image_width: int,
        phy_cell_kernel_size: Tuple[int, int],
        conv_cell_kernel_size: Tuple[int, int],
        k2m_shape: List[int],
        constraints: torch.Tensor,
        loss_function: nn.Module,
    ) -> None:
        super().__init__()

        patch_height: int = image_height // patch_size
        patch_width: int = image_width // patch_size
        frame_channel: int = patch_size * patch_size * img_channel

        self.phycell = PhyCell(
            # input_shape=(patch_height, patch_width),
            input_shape=(patch_height // 4, patch_width // 4),
            # input_dim=64,
            # in_channels=frame_channel,
            in_channels=64,
            # F_hidden_dims=[49],
            F_hidden_dims=phy_cell_num_hidden,
            # n_layers=1,
            n_layers=len(phy_cell_num_hidden),
            # kernel_size=(7, 7),
            kernel_size=phy_cell_kernel_size,
        )
        self.convcell = PhyD_ConvLSTM(
            # input_shape=(patch_height, patch_width),
            input_shape=(patch_height // 4, patch_width // 4),
            # input_dim=64,
            # in_channels=frame_channel,
            in_channels=64,
            # hidden_dims=[128, 128, 64],
            hidden_dims=conv_num_hidden,
            # n_layers=3,
            n_layers=len(conv_num_hidden),
            # kernel_size=(3, 3),
            kernel_size=conv_cell_kernel_size,
        )
        self.encoder = PhyD_EncoderRNN(
            self.phycell, self.convcell, in_channel=frame_channel, patch_size=patch_size
        )
        # self.k2m = K2M([7, 7])
        self.k2m = K2M(k2m_shape)

        self.constraints = constraints
        self.loss_function = loss_function()

    def forward(
        self,
        frames_tensor: torch.Tensor,
        mask_true: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, total_length, _, height, width = frames_tensor.shape
        loss = 0.0

        generated_frames: List[torch.Tensor] = []
        generated_frame = None
        for t in range(total_length - 1):
            cellInput = blend_input_frame(frames_tensor, mask_true, t, generated_frame)

            _, _, generated_frame, _, _ = self.encoder(cellInput, (t == 0))

            target = frames_tensor[:, t + 1, :, :, :]
            loss += self.loss_function(generated_frame, target)

            generated_frames.append(generated_frame)

        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
            m = self.k2m(filters.double()).float().to(self.constraints.device)
            loss += self.loss_function(m, self.constraints)

        targetLength = mask_true.size(1) + 1
        output = (
            torch.stack(generated_frames[-targetLength:], 0)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )

        return output, loss
