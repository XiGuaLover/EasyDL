from functools import lru_cache
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from utils.ConfigType import MetricType, PhyDNetConfig
from utils.ScheduledSampler import ScheduledSampler
from utils.tools import (
    calculateMetric,
    calculateMetricOpenSTLStyle,
    calculateSSIM,
    getFinalOptimizers,
    getScheduler,
    reshape_from_patches,
    reshape_to_patches,
    saveMetrics,
    saveTestingImages,
)

from ..cores.PhyDNet import PhyDNet


class LightningPhyDNet(pl.LightningModule):
    def __init__(self, config: PhyDNetConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.lr = config.learning_rate
        self.scheduledSampler = ScheduledSampler(config=config.scheduledSampleConfig)
        self.criterion = config.experimentConfig.criterionFunction()
        self.validation_step_outputs = []
        self.metricsOpenSTLStyle = []

        constraints = (
            torch.eye(config.constraints_shape[0])
            .reshape(config.constraints_shape)
            .to(self.device)
        )

        self.model = PhyDNet(
            phy_cell_num_hidden=config.phy_cell_num_hidden,
            conv_num_hidden=config.conv_num_hidden,
            phy_cell_kernel_size=config.phy_cell_kernel_size,
            conv_cell_kernel_size=config.conv_cell_kernel_size,
            patch_size=config.scheduledSampleConfig.patch_size,
            img_channel=config.img_channel,
            image_height=config.img_height,
            image_width=config.img_width,
            k2m_shape=config.k2m_shape,
            constraints=constraints,
            loss_function=config.experimentConfig.criterionFunction,
        )

    def forward(
        self,
        batchImageInput: Tensor,
        batchImageTarget: Tensor,
        stage: str,
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
        outputImages, decouple_loss = self.model(images, sampling_mask_tensor)
        self.decouple_loss = decouple_loss

        return reshape_from_patches(
            outputImages, self.config.scheduledSampleConfig.patch_size
        )

    def _compute_loss(
        self, batchImageTarget: Tensor, batchOutputImages: Tensor
    ) -> Tensor:
        return self.criterion(batchImageTarget, batchOutputImages)

    def _shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> Tensor:
        batchImageInput, batchImageTarget = batch
        outputImages = self(batchImageInput, batchImageTarget, stage)
        loss = self._compute_loss(batchImageTarget, outputImages)

        if stage == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.log(
                "validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True
            )

            result = {}
            resultOpenSTLStyle = {}
            for metric in self.config.experimentConfig.metrics:
                result[metric] = calculateMetric(metric, outputImages, batchImageTarget)
                resultOpenSTLStyle[metric] = calculateMetricOpenSTLStyle(
                    metric, outputImages, batchImageTarget
                )
                if metric == MetricType.SSIM:
                    _, seqLen, _, _, _ = outputImages.shape
                    for t in range(seqLen):
                        y_hat_frames = outputImages[:, t, :, :, :]
                        y_frames = batchImageTarget[:, t, :, :, :]
                        resultOpenSTLStyle[metric.value + "_" + str(t)] = calculateSSIM(
                            y_hat_frames, y_frames
                        )
            result["validation_loss"] = loss
            self.validation_step_outputs.append(result)
            self.metricsOpenSTLStyle.append(resultOpenSTLStyle)

            return result

        if stage == "test":
            self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            if batch_idx < self.config.num_save_samples:
                saveTestingImages(
                    savePath=self.config.images_save_dir,
                    patchedX=batchImageInput,
                    patchedY=batchImageTarget,
                    patchedYHat=outputImages,
                    patchSize=1,
                    batchIndex=batch_idx,
                    epoch=self.current_epoch,
                )

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> float:
        return self._shared_step(batch, batch_idx, "test")

    def on_validation_epoch_end(self):
        saveMetrics(
            self.validation_step_outputs,
            current_epoch=self.current_epoch,
            writeTo=self._get_name(),
        )
        saveMetrics(
            self.metricsOpenSTLStyle,
            current_epoch=self.current_epoch,
            writeTo=self._get_name(),
        )

    def configure_optimizers(self):
        optimizer: Optimizer = self.config.experimentConfig.optimizer(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = getScheduler(
            optimizer,
            self.lr,
            self.trainer,
            self.config.experimentConfig.schedulerConfig,
        )

        return getFinalOptimizers(optimizer, scheduler)

    def _batchToPatch(
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, int, int, int, int]:
        batchSize, targetLength, channels, height, width = y.shape
        patch_size = self.config.scheduledSampleConfig.patch_size
        patchX = reshape_to_patches(x, patch_size)
        patchY = reshape_to_patches(y, patch_size)
        patched_height = height // patch_size
        patched_width = width // patch_size
        patchChannel = channels * (patch_size**2)
        return patchX, patchY, targetLength, patched_height, patched_width, patchChannel

    @lru_cache(maxsize=32)
    def _get_cached_mask(
        self,
        batch_size: int,
        target_length: int,
        channel: int,
        height: int,
        width: int,
        is_training: bool,
    ) -> np.ndarray:
        if is_training:
            return self.scheduledSampler.training_mask(
                training_step=self.global_step,
                batchSize=batch_size,
                targetLength=target_length,
                channel=channel,
                height=height,
                width=width,
            )
        return self.scheduledSampler.test_mask(
            batchSize=batch_size,
            targetLength=target_length,
            channel=channel,
            height=height,
            width=width,
        )
