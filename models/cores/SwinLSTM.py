from typing import List, Tuple

import torch
import torch.nn as nn

from ..components.SwinLSTMComponent import DownSample, STconvert, UpSample


class SwinLSTMDeep(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depths_downSample: List[int],
        depths_upsample: List[int],
        num_heads: List[int],
        window_size: int,
    ) -> None:
        super().__init__()
        self.depths_downSample: List[int] = depths_downSample
        self.depths_upSample: List[int] = depths_upsample
        self.downSample: DownSample = DownSample(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths_downSample=depths_downSample,
            num_heads=num_heads,
            window_size=window_size,
        )

        self.Upsample: UpSample = UpSample(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths_upsample=depths_upsample,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(self, inputImages: torch.Tensor, targetLength: int) -> torch.Tensor:
        batch_size, seqLength, _, height, width = inputImages.shape

        states_down: List[torch.Tensor | None] = [None] * len(self.depths_downSample)
        states_up: List[torch.Tensor | None] = [None] * len(self.depths_upSample)

        generated_frames: List[torch.Tensor] = []
        last_frame: torch.Tensor = inputImages[:, -1]

        for t in range(seqLength - 1):
            states_down, x = self.downSample(inputImages[:, t], states_down)
            states_up, output = self.Upsample(x, states_up)
            generated_frames.append(output)

        for t in range(targetLength):
            states_down, x = self.downSample(last_frame, states_down)
            states_up, output = self.Upsample(x, states_up)
            generated_frames.append(output)
            last_frame = output

        output_frames: torch.Tensor = (
            torch.stack(generated_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        )

        return output_frames


class SwinLSTMBase(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        in_channels: int,
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: int,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()

        self.ST: STconvert = STconvert(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, inputImages: torch.Tensor, targetLength: int) -> torch.Tensor:
        batch_size, seqLength, _, height, width = inputImages.shape

        states: torch.Tensor | None = None
        generated_frames: List[torch.Tensor] = []
        last_frame: torch.Tensor = inputImages[:, -1]

        for t in range(seqLength - 1):
            output: torch.Tensor
            output, states = self.ST(inputImages[:, t], states)
            generated_frames.append(output)

        for t in range(targetLength):
            output: torch.Tensor
            output, states = self.ST(last_frame, states)
            generated_frames.append(output)
            last_frame = output

        output_frames: torch.Tensor = (
            torch.stack(generated_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        )

        return output_frames
