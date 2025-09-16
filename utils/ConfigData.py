from pathlib import Path
from typing import List

import torch
from libs.pyphoon2.DigitalTyphoonUtils import LOAD_DATA, SPLIT_UNIT, TRACK_COLS

from .configDatas.defaultConfigData import defaultDataPathConfig
from .ConfigType import (
    BaseRNNConfig,
    ConvLSTMConfig,
    CosineAnnealingLRConfig,
    DataModuleID,
    DataSetTimeSetting,
    DigitalTyphoonTimeSeriesDataLoadConfig,
    ExperimentConfig,
    LightningModelID,
    MetricType,
    NetConfig,
    NetID,
    OneCycleLRConfig,
    PredRNNConfig,
    PredRnnV2Config,
    RunConfig,
    ScheduledSampleConfig,
)


class ModelCheckPointConfigData:
    filenamePrefix = "model"
    monitor = "validation_loss"
    mode = "min"
    save_top_k = 3
    patience = 5


class EarlyStopConfigData:
    monitor = "validation_loss"
    mode = "min"
    patience = 5


class SuperParams:
    runNets: List[NetID] = [
        NetID.ConvLSTM_DigitalTyphoon,
        NetID.PredRnn_DigitalTyphoon,
        NetID.PredRnnV2_DigitalTyphoon,
        NetID.PredRnnPP_DigitalTyphoon,
        NetID.ConvLSTM_DigitalTyphoonOne,
        NetID.PredRnn_DigitalTyphoonOne,
        NetID.PredRnnV2_DigitalTyphoonOne,
        NetID.PredRnnPP_DigitalTyphoonOne,
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logDir = Path("./logs/")
    loggerSaveDir = "logger"
    metricsLogDir = "metrics"
    foundLrDir = "foundLr"


class NetConfigData:
    data: List[NetConfig] = [
        NetConfig(
            id=NetID.ConvLSTM_DigitalTyphoon,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningConvLSTM,
            lightningModelConfig=ConvLSTMConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_ConvLSTM/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=CosineAnnealingLRConfig(
                            T_max=10,
                            eta_min=1e-6,
                        ),
                    ),
                )
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnn_DigitalTyphoon,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRNN,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRNN/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=CosineAnnealingLRConfig(
                            T_max=10,
                            eta_min=1e-6,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnnV2_DigitalTyphoon,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnV2,
            lightningModelConfig=PredRnnV2Config(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRnnV2/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=CosineAnnealingLRConfig(
                            T_max=10,
                            eta_min=1e-6,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
                decouple_beta=0.1,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnnPP_DigitalTyphoon,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnPP,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRnnPP/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=CosineAnnealingLRConfig(
                            T_max=10,
                            eta_min=1e-6,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.ConvLSTM_DigitalTyphoonOne,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningConvLSTM,
            lightningModelConfig=ConvLSTMConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_ConvLSTM/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=OneCycleLRConfig(
                            pct_start=0.3,
                            div_factor=10,
                            final_div_factor=10000,
                        ),
                    ),
                )
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnn_DigitalTyphoonOne,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRNN,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRNN/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=OneCycleLRConfig(
                            pct_start=0.3,
                            div_factor=10,
                            final_div_factor=10000,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnnV2_DigitalTyphoonOne,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnV2,
            lightningModelConfig=PredRnnV2Config(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRnnV2/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=OneCycleLRConfig(
                            pct_start=0.3,
                            div_factor=10,
                            final_div_factor=10000,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
                decouple_beta=0.1,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
        NetConfig(
            id=NetID.PredRnnPP_DigitalTyphoonOne,
            trainEpoch=200,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnPP,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=BaseRNNConfig(
                    learning_rate=5e-6,
                    num_hidden=[128, 128, 128, 128],
                    images_save_dir="results_test/DigitalTyphoon_PredRnnPP/",
                    img_channel=1,
                    img_height=128,
                    img_width=128,
                    stride=1,
                    kernel_size=(5, 5),
                    num_save_samples=10,
                    scheduledSampleConfig=ScheduledSampleConfig(
                        initial_sampling_rate=1.0,
                        enable_scheduled_sampling=True,
                        # stop iteration / batchSize
                        stop_sampling_training_global_step=int(50000 / 4),
                        sampling_decay_rate=0.00002,
                        patch_size=4,
                    ),
                    experimentConfig=ExperimentConfig(
                        metrics=[
                            MetricType.MAE,
                            MetricType.MSE,
                            MetricType.SSIM,
                            MetricType.RMSE,
                        ],
                        optimizer=torch.optim.AdamW,
                        schedulerConfig=OneCycleLRConfig(
                            pct_start=0.3,
                            div_factor=10,
                            final_div_factor=10000,
                        ),
                    ),
                ),
                patch_size=4,
                useLayerNorm=False,
            ),
            runConfig=RunConfig(
                dataloadConfig=DigitalTyphoonTimeSeriesDataLoadConfig(
                    batchSize=4,
                    labels=[TRACK_COLS.PRESSURE.name.lower()],
                    splitBy=SPLIT_UNIT.SEQUENCE,
                    loadData=LOAD_DATA.ALL_DATA,
                    datasetSplitByTime=DataSetTimeSetting(
                        trainTime=(2015, 2021),
                        valTime=(2022, 2022),
                        testTime=(2023, 2023),
                    ),
                    standardizeRange=(170, 300),
                    downSampleSize=(128, 128),
                    cropped=True,
                    numWorkers=32,
                    pathConfig=defaultDataPathConfig.get(DataModuleID.DigitalTyphoon),
                    inputSeqLen=12,
                    targetSeqLen=12,
                    loadImgsIntoMemory=True,
                    pin_memory=True,
                ),
                saveCheckpoint=True,
            ),
        ),
    ]

    @staticmethod
    def getNetConfig(id: NetID) -> NetConfig:
        for model in NetConfigData.data:
            if model.id == id:
                return model
        raise ValueError(f"Model {id} not found")

    @staticmethod
    def getLabelsString(labels: List[str]) -> str:
        return "-".join([label for label in labels])

    @staticmethod
    def validateUniqueIds():
        ids = [net.id for net in NetConfigData.data]
        unique_ids = set(ids)
        if len(ids) != len(unique_ids):
            duplicates = [id for id in unique_ids if ids.count(id) > 1]
            raise ValueError(f"Warming! duplicate ids found: {duplicates}")
        return True


NetConfigData.validateUniqueIds()
