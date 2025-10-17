from dataclasses import asdict
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
from torch import Tensor
from torch.optim.optimizer import Optimizer
from utils.ConfigType import MetricType, PredFormerConfig
from utils.tools import (
    calculateMetric,
    calculateMetricOpenSTLStyle,
    calculateSSIM,
    getFinalOptimizers,
    getScheduler,
    saveMetrics,
    saveTestingImages,
    writeModelInfoToMetricsFile,
)

from ..cores.PredFormer import PredFormer


class LightningPredFormer(pl.LightningModule):
    def __init__(self, config: PredFormerConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        writeModelInfoToMetricsFile(
            config.learning_rate, config.experimentConfig, self._get_name()
        )
        self.config = config

        self.lr = config.learning_rate
        self.criterion = config.experimentConfig.criterionFunction()
        self.validation_step_outputs = []
        self.metricsOpenSTLStyle = []

        self.model = PredFormer(
            componentConfig=config.componentConfig,
        )

    def forward(self, batchImageInput: Tensor, batchImageTarget: Tensor) -> Tensor:
        return self.model(batchImageInput, batchImageTarget)

    def _shared_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, stage: str
    ) -> Union[Tensor, Dict]:
        batchImageInput, batchImageTarget = batch
        outputImages = self(batchImageInput, batchImageTarget)
        loss = self.criterion(batchImageTarget, outputImages)

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

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        self.last_global_step = checkpoint.get("global_step", -1)
        self.sch_dict = checkpoint["lr_schedulers"][0]
        self.opt_dict = checkpoint["optimizer_states"][0]

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

        # resume from checkpoint
        if self.trainer.ckpt_path is not None and self.last_global_step > 0:
            self.sch_dict["total_steps"] = self.trainer.estimated_stepping_batches
            self.sch_dict["last_epoch"] = self.last_global_step
            if scheduler is not None:
                scheduler.get("scheduler").load_state_dict(self.sch_dict)
            optimizer.load_state_dict(self.opt_dict)
            print(
                f"resume scheduler ={self.last_global_step}, lr={scheduler.get_last_lr()}"
            )

        return getFinalOptimizers(optimizer, scheduler)
