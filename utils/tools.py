import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.tuner import Tuner
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import structural_similarity_index_measure as ssim

from utils.ConfigType import MetricType

from .ConfigData import (
    SuperParams,
)
from .ConfigType import (
    CosineAnnealingLRConfig,
    ExperimentConfig,
    NetConfig,
    OneCycleLRConfig,
)


def getLoggerName(cfg: NetConfig) -> str:
    parts = [
        cfg.id.value,
    ]
    return "_".join(parts)


def getLoggerSaveDir() -> str:
    return os.path.join(SuperParams.logDir, SuperParams.loggerSaveDir)


def check_memory(info: str, device="cuda:0"):
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # 转换为 MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # 转换为 MB
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB", info)
    return allocated, reserved


def reshape_to_patches(img_tensor: Tensor, patch_size: int) -> Tensor:
    batch, seq_length, channels, height, width = img_tensor.shape
    patched_height: int = height // patch_size
    patched_width: int = width // patch_size
    patch_tensor: Tensor = img_tensor.view(
        batch,
        seq_length,
        channels,
        patched_height,
        patch_size,
        patched_width,
        patch_size,
    )
    patch_tensor = patch_tensor.permute(0, 1, 2, 4, 6, 3, 5)
    patch_tensor = patch_tensor.contiguous().view(
        batch,
        seq_length,
        patch_size * patch_size * channels,
        patched_height,
        patched_width,
    )
    return patch_tensor


def reshape_from_patches(patch_tensor: Tensor, patch_size: int) -> Tensor:
    batch, seq_length, channels, patch_h, patch_w = patch_tensor.shape
    img_channels: int = channels // (patch_size * patch_size)
    height = patch_h * patch_size
    width = patch_w * patch_size

    img_tensor = patch_tensor.view(
        batch, seq_length, img_channels, patch_size, patch_size, patch_h, patch_w
    )

    # Reverse the permutation
    img_tensor = img_tensor.permute(0, 1, 2, 5, 3, 6, 4)
    # Reshape to the original image dimensions
    img_tensor = img_tensor.contiguous().view(
        batch, seq_length, img_channels, height, width
    )
    return img_tensor


def blend_input_frame(
    frames: torch.Tensor,
    mask_true: torch.Tensor,
    t: int,
    generated_frame: Optional[torch.Tensor],
) -> torch.Tensor:
    totalLength = frames.size(1)
    targetLength = mask_true.size(1) + 1
    inputLength = totalLength - targetLength

    return (
        frames[:, t]
        if t < inputLength
        else (
            mask_true[:, t - inputLength] * frames[:, t]
            + (1 - mask_true[:, t - inputLength]) * generated_frame
        )
    )


def saveTestingImages(
    savePath: str,
    patchedX: Tensor,
    patchedY: Tensor,
    patchedYHat: Tensor,
    patchSize: int,
    batchIndex: int = 0,
    epoch: int = -1,
) -> None:
    patchedYHat = reshape_from_patches(patchedYHat, patchSize)

    # Process first sample in batch
    draw_x = patchedX[0].cpu().numpy().squeeze(1)
    draw_y = patchedY[0].cpu().numpy().squeeze(1)
    draw_y_hat = patchedYHat[0].cpu().numpy().squeeze(1)
    colNums = max(len(draw_x), len(draw_y), len(draw_y_hat))
    fig, axes = plt.subplots(nrows=3, ncols=colNums, figsize=(2 * colNums, 4))

    axes = np.atleast_2d(axes)
    for i in range(colNums):
        for row, frames in enumerate([draw_x, draw_y, draw_y_hat]):
            if i < len(frames):
                axes[row, i].imshow(frames[i], cmap="gray")
                axes[row, i].axis("off")
    plt.tight_layout()

    epoch_dir = os.path.join(savePath, f"epoch-{epoch}")
    results_dir: str = os.path.join(
        epoch_dir,
        datetime.now().strftime("%Y-%m-%d_%H_%M"),
        f"inputLen_{len(draw_x)}_outputLen_{len(draw_y)}_predictedLen_{len(draw_y_hat)}",
    )

    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"batchIndex_{batchIndex}.png"))
    plt.close(fig)


def saveTestingOriginImages(
    savePath: str,
    inputImages: Tensor,
    targetImages: Tensor,
    outputImages: Tensor,
    batchIndex: int = 0,
    epoch: int = -1,
) -> None:
    # Process first sample in batch
    draw_x = inputImages[0].cpu().numpy().squeeze(1)
    draw_y = targetImages[0].cpu().numpy().squeeze(1)
    draw_y_hat = outputImages[0].cpu().numpy().squeeze(1)
    colNums = max(len(draw_x), len(draw_y), len(draw_y_hat))
    fig, axes = plt.subplots(nrows=3, ncols=colNums, figsize=(2 * colNums, 4))

    axes = np.atleast_2d(axes)
    for i in range(colNums):
        for row, frames in enumerate([draw_x, draw_y, draw_y_hat]):
            if i < len(frames):
                axes[row, i].imshow(frames[i], cmap="gray")
                axes[row, i].axis("off")
    plt.tight_layout()

    epoch_dir = os.path.join(savePath, f"epoch-{epoch}")
    results_dir: str = os.path.join(
        epoch_dir,
        datetime.now().strftime("%Y-%m-%d_%H_%M"),
        f"inputLen_{len(draw_x)}_outputLen_{len(draw_y)}_predictedLen_{len(draw_y_hat)}",
    )

    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"batchIndex_{batchIndex}.png"))
    plt.close(fig)


def saveTestingLabels(
    modelName: str,
    targetLabels: Tensor,
    outputLabels: Tensor,
    epoch: int,
) -> None:
    log_file = os.path.join(SuperParams.logDir, "testingLabels.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Model: {modelName} epoch: {epoch}\n")
        f.write(f"[targetLabels: {targetLabels}\n")
        f.write(f"[outputLabels: {outputLabels}\n")
        f.write("\n")


def getScheduler(
    optimizer: Optimizer,
    lr: float,
    trainer: pl.Trainer,
    schedulerConfig: Optional[Union[OneCycleLRConfig, CosineAnnealingLRConfig]],
) -> Optional[Dict]:
    if schedulerConfig is None:
        return None
    if isinstance(schedulerConfig, OneCycleLRConfig):
        total_steps = trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=schedulerConfig.pct_start,
            div_factor=schedulerConfig.div_factor,
            final_div_factor=schedulerConfig.final_div_factor,
        )
        return dict(
            {
                "scheduler": scheduler,
                "interval": "step",
            },
        )

    if isinstance(schedulerConfig, CosineAnnealingLRConfig):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=schedulerConfig.T_max,
            eta_min=schedulerConfig.eta_min,
        )
        return dict(
            {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        )

    return None


def getFinalOptimizers(optimizer: Optimizer, scheduler: Optional[Dict]):
    if scheduler is None:
        return optimizer
    else:
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def calculateLabelMetric(metricType: MetricType, y_hat: Tensor, y: Tensor):
    if y_hat.dim() == y.dim() + 1 and y_hat.size(-1) == 1:
        y_hat = y_hat.squeeze(-1)

    if metricType == MetricType.MAE:
        return torch.nn.L1Loss()(y_hat, y)
    elif metricType == MetricType.MSE:
        return torch.nn.MSELoss()(y_hat, y)
    elif metricType == MetricType.RMSE:
        return torch.sqrt(torch.nn.MSELoss()(y_hat, y))

    else:
        raise NotImplementedError("Metric type not implemented")


def calculateMetric(metricType: MetricType, y_hat: Tensor, y: Tensor):
    if metricType == MetricType.MAE:
        return torch.nn.L1Loss(reduction="mean")(y_hat, y)

    elif metricType == MetricType.MSE:
        return torch.nn.MSELoss(reduction="mean")(y_hat, y)

    elif metricType == MetricType.RMSE:
        return torch.sqrt(torch.nn.MSELoss(reduction="mean")(y_hat, y))
    elif metricType == MetricType.SSIM:
        ssim_scores = []
        _, seqLen, _, _, _ = y_hat.shape
        for t in range(seqLen):
            # Extract frame t: shape (batch_size, channels, height, width)
            y_hat_frame = y_hat[:, t, :, :, :]
            y_frame = y[:, t, :, :, :]
            # Compute SSIM for the frame
            score = ssim(y_hat_frame, y_frame, data_range=1.0)
            ssim_scores.append(score)
        result = torch.mean(torch.stack(ssim_scores))
        return result

    else:
        raise NotImplementedError("Metric type not implemented")


def calculateMetricOpenSTLStyle(metricType: MetricType, y_hat: Tensor, y: Tensor):
    if metricType == MetricType.MAE:
        abs_diff = torch.abs(y_hat - y)
        mean_abs_diff = torch.mean(abs_diff, dim=(0, 1))
        result = torch.sum(mean_abs_diff)
        return result

        # return torch.nn.L1Loss()(y_hat, y)
    elif metricType == MetricType.MSE:
        abs_diff = torch.abs(y_hat - y) ** 2
        mean_abs_diff = torch.mean(abs_diff, dim=(0, 1))
        result = torch.sum(mean_abs_diff)
        return result
    elif metricType == MetricType.RMSE:
        abs_diff = torch.abs(y_hat - y) ** 2
        mean_abs_diff = torch.mean(abs_diff, dim=(0, 1))
        result = torch.sqrt(torch.sum(mean_abs_diff))
        return result
    elif metricType == MetricType.SSIM:
        ssim_scores = []
        _, seqLen, _, _, _ = y_hat.shape
        for t in range(seqLen):
            y_hat_frame = y_hat[:, t, :, :, :]
            y_frame = y[:, t, :, :, :]
            score = ssim(y_hat_frame, y_frame, data_range=1.0)
            ssim_scores.append(score)
        result = torch.mean(torch.stack(ssim_scores))
        return result

    else:
        raise NotImplementedError("Metric type not implemented")


def calculateSSIM(y_hat_frames: Tensor, y_frames: Tensor):
    score = ssim(y_hat_frames, y_frames, data_range=1.0)
    result = torch.mean(score)
    return result


def geMetricsLoggerDir(writeTo: str) -> str:
    dir = os.path.join(SuperParams.logDir, SuperParams.metricsLogDir)
    os.makedirs(dir, exist_ok=True)
    fileName = f"{writeTo}_metrics.txt"
    return os.path.join(dir, fileName)


def writeModelInfoToMetricsFile(
    lr: float,
    expConfig: ExperimentConfig,
    writeTo: str = "unknownClass",
):
    fileName = geMetricsLoggerDir(writeTo)
    with open(fileName, "a") as f:
        f.write(
            f"time: {str(datetime.now().strftime('%Y%m%d-%H.%M.%S'))},lr: {lr} , expConfig: {expConfig}\n"
        )


def writeMetrics(
    metricType: str, value: float, current_epoch: int, writeTo: str = "unknownClass"
):
    fileName = geMetricsLoggerDir(writeTo)
    with open(fileName, "a") as f:
        f.write(
            f"time: {str(datetime.now().strftime('%Y%m%d-%H.%M.%S'))},epoch: {current_epoch} , metricType: {metricType}, value: {value:.5f}\n"
        )


def saveMetrics(metrics_list: List, current_epoch: int, writeTo: str = "unknownClass"):
    if metrics_list:
        resultKeys = metrics_list[0].keys()
        for key in resultKeys:
            loss = torch.stack([x[key] for x in metrics_list])
            loss = loss.mean()
            writeMetrics(
                metricType=key,
                value=loss,
                current_epoch=current_epoch,
                writeTo=writeTo,
            )

        metrics_list.clear()


def writeFLOPs(modelName: str, summary: str):
    log_file = os.path.join(SuperParams.logDir, "flops.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] Model: {modelName} FLOPs and Parameters Summary \n")
        f.write(summary)
        f.write("\n")


def findLR(
    trainer: pl.Trainer, data_module: pl.LightningDataModule, model: pl.LightningModule
):
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model, datamodule=data_module, num_training=1000, min_lr=1e-6
    )

    dir = os.path.join(SuperParams.logDir, SuperParams.foundLrDir)
    os.makedirs(dir, exist_ok=True)

    # Plot
    modelName = model._get_name()
    time = datetime.now().strftime("%Y%m%d-%H.%M.%S")
    fig = lr_finder.plot(suggest=True)
    figPath = os.path.join(dir, f"lr_tune_{modelName}_{time}.png")
    fig.savefig(figPath)

    foundLrLogFile = os.path.join(dir, "lr_tune.log")
    with open(foundLrLogFile, "a") as f:
        f.write(
            f"time: {time}, modelName:{modelName}, lr_finder.suggestion(): {lr_finder.suggestion()}\n"
        )
    print("lr_finder.suggestion()!!!!!", lr_finder.suggestion())
    return lr_finder.suggestion()
