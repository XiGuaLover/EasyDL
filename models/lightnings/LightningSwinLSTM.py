from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from utils.ConfigType import MetricType, SwinLSTMConfig
from utils.tools import (
    calculateMetric,
    calculateMetricOpenSTLStyle,
    calculateSSIM,
    getFinalOptimizers,
    getScheduler,
    saveMetrics,
    saveTestingImages,
)

from ..cores.SwinLSTM import SwinLSTMDeep


class LightningSwinLSTM(pl.LightningModule):
    def __init__(self, config: SwinLSTMConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.lr = config.learning_rate
        self.criterion = config.experimentConfig.criterionFunction()
        self.validation_step_outputs = []
        self.metricsOpenSTLStyle = []

        self.model = SwinLSTMDeep(
            img_size=config.input_img_size,
            patch_size=config.patch_size,
            in_channels=config.input_channels,
            embed_dim=config.embed_dim,
            depths_downSample=config.depths_downSample,
            depths_upsample=config.depths_upsample,
            num_heads=config.heads_number,
            window_size=config.window_size,
        )

    def forward(self, batchImageInput: Tensor, batchImageTarget: Tensor) -> Tensor:
        return self.model(batchImageInput, batchImageTarget.size(1))

    def _shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> Union[Tensor, Dict]:
        batchImageInput, batchImageTarget = batch
        outputImages = self(batchImageInput, batchImageTarget)
        target = torch.cat((batchImageInput[:, 1:], batchImageTarget), dim=1)
        loss = self.criterion(target, outputImages)

        if stage == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if stage == "val":
            self.log(
                "validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True
            )

            result = {}
            resultOpenSTLStyle = {}
            for metric in self.config.experimentConfig.metrics:
                result[metric] = calculateMetric(metric, outputImages, target)
                resultOpenSTLStyle[metric] = calculateMetricOpenSTLStyle(
                    metric, outputImages, target
                )
                if metric == MetricType.SSIM:
                    _, seqLen, _, _, _ = outputImages.shape
                    for t in range(seqLen):
                        y_hat_frames = outputImages[:, t, :, :, :]
                        y_frames = target[:, t, :, :, :]
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

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict:
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
