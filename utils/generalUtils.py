import os
from datetime import datetime
from typing import List, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.TyphoonTimeSeriesModule import TyphoonTimeSeriesModule
from models.lightnings.LightningConvLSTM import LightningConvLSTM
from models.lightnings.LightningE3DLSTM import LightningE3DLSTM
from models.lightnings.LightningPhyDNet import LightningPhyDNet
from models.lightnings.LightningPredFormer import LightningPredFormer
from models.lightnings.LightningPredRNN import LightningPredRNN
from models.lightnings.LightningPredRnnPP import LightningPredRnnPP
from models.lightnings.LightningPredRnnV2 import LightningPredRnnV2
from models.lightnings.LightningSwinLSTM import LightningSwinLSTM

from .callbacks import (
    InitLogCallback,
    MemoryMonitorCallback,
    RuntimeCallback,
)
from .ConfigData import EarlyStopConfigData, ModelCheckPointConfigData, SuperParams
from .ConfigType import (
    DigitalTyphoonTimeSeriesDataLoadConfig,
    LightningModelID,
    NetConfig,
)
from .tools import getLoggerName


def setupLogger(
    logger: TensorBoardLogger,
    cfg: NetConfig,
):
    startTime = str(datetime.now().strftime("%Y_%m_%d-%H.%M.%S"))

    logger.log_hyperparams(
        {
            "start_time": startTime,
            "MODEL_NAME": cfg.id,
            "seed": cfg.seed,
            "MAX_EPOCHS": cfg.trainEpoch,
            "ACCELERATOR": SuperParams.accelerator,
            "specifyDevices": cfg.specifyDevices,
            "model_config": cfg.lightningModelConfig,
        }
    )


def getTyphoonTimeSeriesDataModule(
    cfg: DigitalTyphoonTimeSeriesDataLoadConfig,
) -> TyphoonTimeSeriesModule:
    dataModule = TyphoonTimeSeriesModule(
        dataPathConfig=cfg.pathConfig,
        batch_size=cfg.batchSize,
        # num_workers=cfg.runConfig.numWorkers,
        labels=cfg.labels,
        # no split_by=cfg.splitBy.name.lower(),
        num_workers=cfg.numWorkers,
        load_data=cfg.loadData.name.lower(),
        datasetSplitByRatio=cfg.datasetSplitByRatio,
        datasetSplitByTime=cfg.datasetSplitByTime,
        standardize_range=cfg.standardizeRange,
        downSample_size=cfg.downSampleSize,
        cropped=cfg.cropped,
        inputSeqLen=cfg.inputSeqLen,
        targetSeqLen=cfg.targetSeqLen,
        returnLabels=cfg.returnLabels,
        loadImgsIntoMemory=cfg.loadImgsIntoMemory,
        pin_memory=cfg.pin_memory,
    )
    return dataModule


def getLightningPredRNN(cfg) -> LightningPredRNN:
    model = LightningPredRNN(
        config=cfg,
    )
    return model


def getLightningPredRnnV2(cfg) -> LightningPredRnnV2:
    model = LightningPredRnnV2(
        config=cfg,
    )
    return model


def getLightningPredRNNPP(cfg) -> LightningPredRnnPP:
    model = LightningPredRnnPP(
        config=cfg,
    )
    return model


def getLightningConvLSTM(cfg) -> LightningConvLSTM:
    model = LightningConvLSTM(
        config=cfg,
    )
    return model


def getLightningE3DLSTM(cfg) -> LightningE3DLSTM:
    model = LightningE3DLSTM(config=cfg)
    return model


def getLightningPredFormer(cfg) -> LightningPredFormer:
    model = LightningPredFormer(config=cfg)
    return model


def getLightningSwinLSTM(cfg) -> LightningSwinLSTM:
    model = LightningSwinLSTM(config=cfg)
    return model


def getLightningPhyDNet(cfg) -> LightningPhyDNet:
    model = LightningPhyDNet(config=cfg)
    return model


def getDataModule(
    cfg: NetConfig,
) -> Union[TyphoonTimeSeriesModule,]:
    if isinstance(cfg.runConfig.dataloadConfig, DigitalTyphoonTimeSeriesDataLoadConfig):
        return getTyphoonTimeSeriesDataModule(cfg=cfg.runConfig.dataloadConfig)


def _getModel(
    lightningModelID: LightningModelID, lightningModelConfig
) -> pl.LightningModule:
    modelRouter = {
        LightningModelID.LightningConvLSTM: getLightningConvLSTM,
        LightningModelID.LightningPredRNN: getLightningPredRNN,
        LightningModelID.LightningPredRnnV2: getLightningPredRnnV2,
        LightningModelID.LightningPredRnnPP: getLightningPredRNNPP,
        LightningModelID.LightningE3DLSTM: getLightningE3DLSTM,
        LightningModelID.LightningPhyDNet: getLightningPhyDNet,
        LightningModelID.LightningPredFormer: getLightningPredFormer,
        LightningModelID.LightningSwinLSTM: getLightningSwinLSTM,
    }

    model = modelRouter.get(lightningModelID)
    if model is None:
        raise ValueError(f"Unknown lightning model id: {lightningModelID}")
    return model(cfg=lightningModelConfig)


def getModel(cfg: NetConfig):
    return _getModel(cfg.lightningModelID, cfg.lightningModelConfig)


def _getModelFromCheckpoint(
    lightningModelID: LightningModelID, checkpoint_path: str, modelCfg
) -> pl.LightningModule:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Map LightningModelID to the corresponding model class
    model_router = {
        LightningModelID.LightningConvLSTM: LightningConvLSTM,
        LightningModelID.LightningPredRNN: LightningPredRNN,
        LightningModelID.LightningPredRnnV2: LightningPredRnnV2,
        LightningModelID.LightningPredRnnPP: LightningPredRnnPP,
        LightningModelID.LightningE3DLSTM: LightningE3DLSTM,
        LightningModelID.LightningPhyDNet: LightningPhyDNet,
        LightningModelID.LightningPredFormer: LightningPredFormer,
        LightningModelID.LightningSwinLSTM: LightningSwinLSTM,
    }

    model_class: pl.LightningModule = model_router.get(lightningModelID)
    if model_class is None:
        raise ValueError(f"Unknown lightning model ID: {lightningModelID}")

    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_path, config=modelCfg
        )
    except Exception as e:
        raise RuntimeError(f"load from {checkpoint_path} failed: {str(e)}")

    # Set the model to evaluation mode
    model.eval()
    return model


def getTrainer(cfg: NetConfig) -> pl.Trainer:
    logger = TensorBoardLogger(
        save_dir=os.path.join(SuperParams.logDir, "logger"),
        name=getLoggerName(cfg=cfg),
        default_hp_metric=False,
    )
    setupLogger(logger, cfg)

    callbacks: List[Callback] = []
    callbacks.append(RuntimeCallback(log_dir="./logs/runtimeLogs/"))
    callbacks.append(
        MemoryMonitorCallback(
            netID=cfg.id, log_file=os.path.join(SuperParams.logDir, "memory_stats.log")
        )
    )
    callbacks.append(
        InitLogCallback(log_file=os.path.join(SuperParams.logDir, "init.log"))
    )
    if cfg.runConfig.saveCheckpoint:
        # Callback for model checkpoint
        checkpoint_filepath = os.path.join(
            logger.save_dir, logger.name, "version_%d" % logger.version, "checkpoints"
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_filepath,
            monitor=ModelCheckPointConfigData.monitor,
            save_top_k=ModelCheckPointConfigData.save_top_k,
            mode=ModelCheckPointConfigData.mode,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    if cfg.runConfig.earlyStop:
        # Callback for early stopping
        early_stop_callback = EarlyStopping(
            monitor=EarlyStopConfigData.monitor,
            patience=EarlyStopConfigData.patience,
            mode=EarlyStopConfigData.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Setting up the lightning trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator=SuperParams.accelerator,
        devices=cfg.specifyDevices,
        max_epochs=cfg.trainEpoch,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        enable_progress_bar=cfg.enable_progress_bar,
        callbacks=callbacks,
        default_root_dir=SuperParams.logDir,
        profiler=cfg.profiler,
    )

    print("=======================")
    print("logger name:", logger.name)
    print("logger save_dir:", logger.save_dir)
    print("logger version:", logger.version)
    print("=======================")

    return trainer
