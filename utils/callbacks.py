import os
import time
import traceback
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
import torch
from models.lightnings.LightningPredRNN import LightningPredRNN
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader


class TimingCallback(Callback):
    def __init__(self):
        self.epoch_start_time = None
        self.train_start_time = None
        self.epoch_durations = []

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()

    def on_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, trainer, pl_module):
        duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(duration)
        trainer.logger.add_scalar(
            "time/epoch_duration", duration, trainer.current_epoch
        )
        print(f"Epoch {trainer.current_epoch} took {duration:.2f}s")

    def on_train_end(self, trainer, pl_module):
        total_duration = time.time() - self.train_start_time
        # avg_epoch_duration = sum(self.epoch_durations) / len(self.epoch_durations)
        trainer.logger.add_scalar("time/total_duration", total_duration, 0)
        # trainer.logger.add_scalar("time/avg_epoch_duration", avg_epoch_duration, 0)
        print(f"Training finished in {total_duration:.2f}s")
        # print(f"Avg epoch duration: {avg_epoch_duration:.2f}s")


class RuntimeCallback(Callback):
    def __init__(self, log_dir="./"):
        super().__init__()
        self.log_dir = log_dir
        self.batch_times = []
        self.epoch_times = []
        # self.batch_start_time = None
        self.epoch_start_time = None
        logFileName = "runtimeLog" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
        self.log_file = os.path.join(log_dir, logFileName)

    def setup(self, trainer, pl_module, stage=None):
        # Create log directory only on rank 0 to avoid race conditions
        if trainer.global_rank == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            # Initialize log file
            with open(self.log_file, "w") as f:
                f.write("Training Runtime Logs\n")
                f.write("====================\n")

        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     # Record start time of batch
    #     self.batch_start_time = time.time()

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     # Calculate and store batch duration, only log on rank 0
    #     batch_duration = time.time() - self.batch_start_time
    #     self.batch_times.append(batch_duration)

    #     if trainer.global_rank == 0:
    #         with open(self.log_file, "a") as f:
    #             f.write(
    #                 f"Epoch {trainer.current_epoch}, Batch {batch_idx}: {batch_duration:.4f} seconds\n"
    #             )

    def on_train_epoch_start(self, trainer, pl_module):
        # Record start time of epoch
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate and store epoch duration, only log on rank 0
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)

        if trainer.global_rank == 0:
            num_batches = len(trainer.train_dataloader)
            avg_batch_time = (
                sum(self.batch_times[-num_batches:]) / num_batches
                if num_batches > 0
                else 0
            )
            with open(self.log_file, "a") as f:
                f.write(
                    f"Epoch {trainer.current_epoch} completed in {epoch_duration:.4f} seconds\n"
                )
                f.write(f"current time: {time.strftime('%Y%m%d-%H%M%S')}")
                f.write(f"Average batch time: {avg_batch_time:.4f} seconds\n")
                f.write("--------------------\n")

    def on_train_end(self, trainer, pl_module):
        # Log final statistics only on rank 0
        if trainer.global_rank == 0 and self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            avg_batch_time = (
                sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
            )
            with open(self.log_file, "a") as f:
                f.write("Training Summary\n")
                f.write(f"Average epoch time: {avg_epoch_time:.4f} seconds\n")
                f.write(f"Average batch time: {avg_batch_time:.4f} seconds\n")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["callback_state"] = {
            "epoch_start_time": self.epoch_start_time,
            "epoch_times": self.epoch_times,
            "batch_times": self.batch_times,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        callback_state = checkpoint.get("callback_state", {})
        self.epoch_start_time = callback_state.get("epoch_start_time", time.time())
        self.epoch_times = callback_state.get("epoch_times", [])
        self.batch_times = callback_state.get("batch_times", [])


class SaveTestImagesAfterEpochCallback(Callback):
    def __init__(self, num_samples: int = 10) -> None:
        self.num_samples: int = num_samples

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: LightningPredRNN
    ) -> None:
        self._save_test_images(trainer, pl_module)

    def _save_test_images(
        self, trainer: pl.Trainer, pl_module: LightningPredRNN
    ) -> None:
        test_dataloader: Optional[DataLoader] = trainer.datamodule.test_dataloader()
        if test_dataloader is None:
            print("Test dataloader is not available. Skipping image saving.")
            return

        pl_module.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx >= self.num_samples:
                    break

                x, y = batch  # x, y: [batch_size, seq_len, in_channels, H, W]
                y_hat: torch.Tensor = pl_module(
                    x
                )  # Shape: [batch_size, seq_len, out_channels, H, W]
                combine_ims: torch.Tensor = torch.cat(
                    (x, y), dim=1
                )  # Shape: [batch_size, seq_len*2, in_channels, H, W]
                combine_ims = torch.cat(
                    (combine_ims, y_hat), dim=1
                )  # Shape: [batch_size, seq_len*3, in_channels, H, W]
                combine_ims = combine_ims.squeeze(
                    dim=2
                )  # Shape: [batch_size, seq_len*3, H, W]
                pl_module.save_sequence(
                    combine_ims[0], trainer.current_epoch, batch_idx, 10
                )

        pl_module.train()


class MemoryMonitorCallback(Callback):
    def __init__(self, netID, log_file="memory_stats.log"):
        super().__init__()
        self.log_file = log_file
        self.netID = netID
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        if not os.path.exists(self.log_file):
            open(self.log_file, "a").close()

    def _log_memory(self, trainer, pl_module, phase, epoch, stage):
        if not torch.cuda.is_available():
            return

        device = pl_module.device
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        max_allocated = torch.cuda.max_memory_allocated(device_idx)
        max_reserved = torch.cuda.max_memory_reserved(device_idx)
        total = torch.cuda.get_device_properties(device_idx).total_memory

        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        max_allocated_gb = max_allocated / (1024**3)
        max_reserved_gb = max_reserved / (1024**3)
        total_gb = total / (1024**3)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(
                f"[{timestamp}] Model: {self.netID}, Phase: {phase}, Epoch: {epoch},Stage: {stage}, Device: cuda:{device_idx}\n"
            )
            f.write(f"  Allocated Memory (GB): {allocated_gb:.2f}\n")
            f.write(f"  Reserved Memory (GB): {reserved_gb:.2f}\n")
            f.write(f"  Max Allocated Memory (GB): {max_allocated_gb:.2f}\n")
            f.write(f"  Max Reserved Memory (GB): {max_reserved_gb:.2f}\n")
            f.write(f"  Total GPU Memory (GB): {total_gb:.2f}\n")
            f.write("\n")

        # Reset peak stats after logging to capture phase-specific peaks
        torch.cuda.reset_peak_memory_stats(device_idx)

    def on_train_epoch_start(self, trainer, pl_module):
        self._log_memory(trainer, pl_module, "training", trainer.current_epoch, "start")

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking is False:
            self._log_memory(
                trainer, pl_module, "validation", trainer.current_epoch, "start"
            )

    def on_test_epoch_start(self, trainer, pl_module):
        self._log_memory(trainer, pl_module, "testing", trainer.current_epoch, "start")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_memory(trainer, pl_module, "training", trainer.current_epoch, "end")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking is False:
            self._log_memory(
                trainer, pl_module, "validation", trainer.current_epoch, "end"
            )

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_memory(trainer, pl_module, "testing", trainer.current_epoch, "end")


# Custom Logger for DeviceStatsMonitor
class DeviceStatsLogger(Logger):
    def __init__(
        self, netID, log_file: str = "device_stats_log.log", log_frequency="epoch"
    ):
        super().__init__()
        self.log_file = log_file
        self.netID = netID
        self.log_frequency = log_frequency  # 'epoch' for epoch-level logging
        self.current_epoch = 0  # Track current epoch

    @property
    def name(self):
        return "DeviceStatsLogger"

    @property
    def version(self):
        return "0.1"

    def log_metrics(self, metrics, step=None):
        # Determine training phase from trainer state
        phase = "unknown"
        if hasattr(self, "trainer") and self.trainer is not None:
            if self.trainer.training:
                phase = "training"
            elif self.trainer.validating:
                phase = "validation"
            elif self.trainer.testing:
                phase = "testing"
        # Only log DeviceStatsMonitor metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(
                f"[{timestamp}] Model: {self.netID}, Phase: {phase}, Step: {step if step is not None else 'N/A'}\n"
            )
            for key, value in metrics.items():
                if (
                    "DeviceStatsMonitor" in key and "bytes" in key
                ):  # Filter memory metrics
                    value_gb = value / (1024**3)  # Convert bytes to GB
                    f.write(f"  {key} (GB): {value_gb:.2f}\n")
                elif "DeviceStatsMonitor" in key:  # Non-byte metrics (e.g., num_ooms)
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_hyperparams(self, params):
        pass

    def set_trainer(self, trainer: pl.Trainer):
        self.trainer = trainer

    def on_train_epoch_end(self, trainer: pl.Trainer):
        if self.log_frequency == "epoch":
            self.current_epoch = trainer.current_epoch
            # Trigger logging of device stats at epoch end
            metrics = trainer.callback_metrics  # Access current metrics
            self.log_metrics(metrics, step=None)

    def on_validation_epoch_end(self, trainer: pl.Trainer):
        if self.log_frequency == "epoch":
            self.current_epoch = trainer.current_epoch
            metrics = trainer.callback_metrics
            self.log_metrics(metrics, step=None)


class InitLogCallback(Callback):
    def __init__(self, log_file="init_info.log"):
        super().__init__()
        self.log_file = log_file
        self.initialized = False

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.initialized and trainer.global_rank == 0:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            # 打开文件以写入初始化信息
            with open(self.log_file, "a") as f:
                # 写入种子信息
                seed = os.environ.get("PL_GLOBAL_SEED", None)
                f.write(f"Seed set to {seed}\n")
                # 写入设备信息
                f.write(
                    f"GPU available: {torch.cuda.is_available()} (cuda), used: {trainer.accelerator}\n"
                )
                f.write("=======================\n")
                # 写入 logger 信息
                f.write(f"logger name: {trainer.logger.name}\n")
                f.write(f"logger save_dir: {trainer.logger.save_dir}\n")
                f.write(f"logger version: {trainer.logger.version}\n")
                f.write("=======================\n")
                f.write("start fitting\n")
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    f.write(
                        f"You are using a CUDA device ('{device_name}') that has Tensor Cores. "
                    )
                # 写入模型摘要
                if trainer.global_rank == 0:  # 仅在主进程写入模型摘要
                    model_summary = str(summarize(pl_module))
                    f.write(model_summary + "\n")
                f.write("*********************** end info \n\n")
            self.initialized = True

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ):
        if trainer.global_rank == 0:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a") as f:
                f.write(
                    f"Error occurred at {torch.utils.data.get_worker_info() or 'main process'}:\n"
                )
                f.write(f"Exception Type: {type(exception).__name__}\n")
                f.write(f"Exception Message: {str(exception)}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc() + "\n")
                f.write("=======================\n")
