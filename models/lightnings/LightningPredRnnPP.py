from dataclasses import asdict

import torch
from torch import Tensor
from utils.ConfigType import PredRNNConfig
from utils.tools import (
    reshape_from_patches,
)

from ..cores.PredRnnPP import PredRnnPP
from .LightningBaseSequence import LightningBaseSequenceModel


class LightningPredRnnPP(LightningBaseSequenceModel):
    def __init__(self, config: PredRNNConfig):
        super().__init__(config.baseRNNConfig)
        self.save_hyperparameters(asdict(config))
        self.config = config

        self.model = PredRnnPP(
            num_hidden=config.baseRNNConfig.num_hidden,
            patch_size=config.patch_size,
            img_channel=config.baseRNNConfig.img_channel,
            image_height=config.baseRNNConfig.img_height,
            image_width=config.baseRNNConfig.img_width,
            kernel_size=config.baseRNNConfig.kernel_size,
            stride=config.baseRNNConfig.stride,
            layer_norm=config.useLayerNorm,
        )

    def forward(
        self, batchImageInput: Tensor, batchImageTarget: Tensor, stage: str = ""
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
