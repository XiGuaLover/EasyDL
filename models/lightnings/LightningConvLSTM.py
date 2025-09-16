from dataclasses import asdict

import torch
from torch import Tensor
from utils.ConfigType import ConvLSTMConfig
from utils.tools import (
    reshape_from_patches,
)

from ..cores.ConvLSTM import ConvLSTM
from .LightningBaseSequence import LightningBaseSequenceModel


class LightningConvLSTM(LightningBaseSequenceModel):
    def __init__(self, config: ConvLSTMConfig):
        super().__init__(config.baseRNNConfig)
        self.save_hyperparameters(asdict(config))

        self.config = config

        self.model = ConvLSTM(
            num_hidden=config.baseRNNConfig.num_hidden,
            kernel_size=config.baseRNNConfig.kernel_size,
            img_channel=config.baseRNNConfig.img_channel,
            patch_size=config.baseRNNConfig.scheduledSampleConfig.patch_size,
            stride=config.baseRNNConfig.stride,
        )

    def forward(
        self,
        batchImageInput: Tensor,
        batchImageTarget: Tensor,
        stage: str = "",
    ) -> Tensor:
        patchX, patchY, targetLength, patched_height, patched_width, patch_channel = (
            self._batchToPatch(batchImageInput, batchImageTarget)
        )
        is_training = stage == "train"
        sampling_mask = self._get_cached_mask(
            batch_size=batchImageInput.shape[0],
            target_length=targetLength,
            channel=patch_channel,
            height=patched_height,
            width=patched_width,
            is_training=is_training,
        )
        sampling_mask_tensor = (
            torch.from_numpy(sampling_mask).float().to(self.device, non_blocking=True)
        )
        images = torch.cat((patchX, patchY), dim=1)
        outputImages = self.model(images, mask_true=sampling_mask_tensor)
        return reshape_from_patches(
            outputImages, self.config.baseRNNConfig.scheduledSampleConfig.patch_size
        )

    def _compute_loss(
        self, batchImageTarget: Tensor, batchOutputImages: Tensor
    ) -> Tensor:
        return self.criterion(batchImageTarget, batchOutputImages)
