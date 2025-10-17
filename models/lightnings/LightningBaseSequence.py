from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import lru_cache
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
from torch import Tensor
from torch.optim.optimizer import Optimizer

from utils.ConfigType import BaseRNNConfig, MetricType
from utils.ScheduledSampler import ScheduledSampler
from utils.tools import (
    calculateMetric,
    calculateMetricOpenSTLStyle,
    calculateSSIM,
    getFinalOptimizers,
    getScheduler,
    reshape_to_patches,
    saveMetrics,
    saveTestingOriginImages,
)


class LightningBaseSequenceModel(pl.LightningModule, ABC):
    def __init__(self, config: BaseRNNConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.__config = config

        self.scheduledSampler = ScheduledSampler(config=config.scheduledSampleConfig)
        self.criterion = config.experimentConfig.criterionFunction()
        self.validation_step_outputs = []
        self.metricsOpenSTLStyle = []

    @abstractmethod
    def forward(
        self,
        batchImageInput: Tensor,
        batchImageTarget: Tensor,
        stage: str,
    ) -> Tensor:
        pass

    @abstractmethod
    def _compute_loss(
        self, batchImageTarget: Tensor, batchOutputImages: Tensor
    ) -> Tensor:
        pass

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
            for metric in self.__config.experimentConfig.metrics:
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
            if batch_idx < self.__config.num_save_samples:
                saveTestingOriginImages(
                    savePath=self.__config.images_save_dir,
                    inputImages=batchImageInput,
                    targetImages=batchImageTarget,
                    outputImages=outputImages,
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
        optimizer: Optimizer = self.__config.experimentConfig.optimizer(
            self.parameters(),
            lr=self.__config.learning_rate,
        )

        scheduler = getScheduler(
            optimizer,
            self.__config.learning_rate,
            self.trainer,
            self.__config.experimentConfig.schedulerConfig,
        )
        return getFinalOptimizers(optimizer, scheduler)

    def _batchToPatch(
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, int, int, int, int]:
        batchSize, targetLength, channels, height, width = y.shape
        patch_size = self.__config.scheduledSampleConfig.patch_size
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
