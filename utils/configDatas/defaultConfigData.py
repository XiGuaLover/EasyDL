from pathlib import Path
from typing import Dict, Union

import torch

from libs.pyphoon2.DigitalTyphoonUtils import LOAD_DATA, SPLIT_UNIT, TRACK_COLS

from ..ConfigType import (
    BaseRNNConfig,
    DataModuleID,
    DataSetTimeSetting,
    DigitalTyphoonDataPathConfig,
    DigitalTyphoonTimeSeriesDataLoadConfig,
    ExperimentConfig,
    MetricType,
    OneCycleLRConfig,
    RunConfig,
    ScheduledSampleConfig,
)

defaultDataPathConfig: Dict[DataModuleID, any] = {
    DataModuleID.DigitalTyphoon: DigitalTyphoonDataPathConfig(
        dataDir=Path("/root/disk/datasets/WP"),
        metadataJsonPath=Path("/root/disk/datasets/WP/metadata.json"),
        metadataDir=Path("/root/disk/datasets/WP/metadata/"),
        imageDir=Path("/root/disk/datasets/WP/image/"),
    )
}


def getDigitalTyphoonTimeSeriesDataLoadConfig(
    batchSize: int = 4,
    labels: list[str] = None,
    splitBy: SPLIT_UNIT = SPLIT_UNIT.SEQUENCE,
    loadData: LOAD_DATA = LOAD_DATA.ALL_DATA,
    datasetSplitByTime: DataSetTimeSetting = None,
    standardizeRange: tuple[int, int] = (170, 350),
    downSampleSize: tuple[int, int] = (128, 128),
    cropped: bool = True,
    numWorkers: int = 32,
    pathConfig: dict = None,
    inputSeqLen: int = 12,
    targetSeqLen: int = 12,
    loadImgsIntoMemory: bool = True,
    pin_memory: bool = True,
) -> DigitalTyphoonTimeSeriesDataLoadConfig:
    if labels is None:
        labels = [TRACK_COLS.PRESSURE.name.lower()]
    if datasetSplitByTime is None:
        datasetSplitByTime = DataSetTimeSetting(
            trainTime=(2015, 2020),
            valTime=(2021, 2022),
            testTime=(2023, 2024),
        )
    if pathConfig is None:
        pathConfig = defaultDataPathConfig.get(DataModuleID.DigitalTyphoon)

    return DigitalTyphoonTimeSeriesDataLoadConfig(
        batchSize=batchSize,
        labels=labels,
        splitBy=splitBy,
        loadData=loadData,
        datasetSplitByTime=datasetSplitByTime,
        standardizeRange=standardizeRange,
        downSampleSize=downSampleSize,
        cropped=cropped,
        numWorkers=numWorkers,
        pathConfig=pathConfig,
        inputSeqLen=inputSeqLen,
        targetSeqLen=targetSeqLen,
        loadImgsIntoMemory=loadImgsIntoMemory,
        pin_memory=pin_memory,
    )


def getExperimentConfig() -> ExperimentConfig:
    return ExperimentConfig(
        metrics=[
            MetricType.MAE,
            MetricType.MSE,
            MetricType.SSIM,
            MetricType.RMSE,
        ],
        optimizer=torch.optim.AdamW,
        schedulerConfig=OneCycleLRConfig(
            final_div_factor=10000.0,
        ),
    )


def getBaseRNNConfig(
    learning_rate: float = 1e-3,
    num_hidden: list[int] = None,
    images_save_dir: str = "results_test/DigitalTyphoon_ConvLSTM/",
    img_channel: int = 1,
    img_height: int = 128,
    img_width: int = 128,
    stride: int = 1,
    kernel_size: Union[tuple[int, int], tuple[int, int, int]] = (5, 5),
    num_save_samples: int = 10,
    scheduledSampleConfig: ScheduledSampleConfig = None,
    experimentConfig: ExperimentConfig = None,
) -> BaseRNNConfig:
    if num_hidden is None:
        num_hidden = [128, 128, 128, 128]
    if scheduledSampleConfig is None:
        scheduledSampleConfig = ScheduledSampleConfig(
            initial_sampling_rate=1.0,
            enable_scheduled_sampling=True,
            # stop iteration / batchSize
            stop_sampling_training_global_step=int(50000 / 4),
            sampling_decay_rate=0.00002,
            patch_size=4,
        )

    if experimentConfig is None:
        experimentConfig = getExperimentConfig()

    return BaseRNNConfig(
        learning_rate=learning_rate,
        num_hidden=num_hidden,
        images_save_dir=images_save_dir,
        img_channel=img_channel,
        img_height=img_height,
        img_width=img_width,
        stride=stride,
        kernel_size=kernel_size,
        num_save_samples=num_save_samples,
        scheduledSampleConfig=scheduledSampleConfig,
        experimentConfig=experimentConfig,
    )


def getDigitalTyphoonRunConfig(
    trainTime: tuple[int, int] = (2015, 2021),
    valTime: tuple[int, int] = (2022, 2022),
    testTime: tuple[int, int] = (2023, 2023),
    standardizeRange: tuple[int, int] = (170, 300),
    saveCheckpoint: bool = True,
    earlyStop: bool = False,
) -> RunConfig:
    dataloadConfig = getDigitalTyphoonTimeSeriesDataLoadConfig(
        datasetSplitByTime=DataSetTimeSetting(
            trainTime=trainTime,
            valTime=valTime,
            testTime=testTime,
        ),
        standardizeRange=standardizeRange,
    )
    return RunConfig(
        dataloadConfig=dataloadConfig,
        saveCheckpoint=saveCheckpoint,
        earlyStop=earlyStop,
    )
