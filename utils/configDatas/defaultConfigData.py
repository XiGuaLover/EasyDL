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
    PhyDNetConfig,
    PredFormerComponentConfig,
    PredFormerConfig,
    RunConfig,
    ScheduledSampleConfig,
    SwinLSTMConfig,
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


def getPhyDNetConfig(
    learning_rate: float = 1e-3,
    images_save_dir: str = "results_test/DigitalTyphoon_PhyDNet/",
    num_save_samples: int = 10,
    img_channel: int = 1,
    img_height: int = 128,
    img_width: int = 128,
    scheduledSampleConfig: ScheduledSampleConfig = None,
    experimentConfig: ExperimentConfig = None,
    phy_cell_num_hidden: list[int] = None,
    conv_num_hidden: list[int] = None,
    phy_cell_kernel_size: tuple[int, int] = (7, 7),
    conv_cell_kernel_size: tuple[int, int] = (3, 3),
    k2m_shape: list[int] = None,
    constraints_shape: tuple[int, int, int] = (49, 7, 7),
) -> PhyDNetConfig:
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

    if phy_cell_num_hidden is None:
        phy_cell_num_hidden = [49]

    if conv_num_hidden is None:
        conv_num_hidden = [128, 128, 64]

    if k2m_shape is None:
        k2m_shape = [7, 7]

    return PhyDNetConfig(
        learning_rate=learning_rate,
        images_save_dir=images_save_dir,
        num_save_samples=num_save_samples,
        img_channel=img_channel,
        img_height=img_height,
        img_width=img_width,
        phy_cell_num_hidden=phy_cell_num_hidden,
        conv_num_hidden=conv_num_hidden,
        phy_cell_kernel_size=phy_cell_kernel_size,
        conv_cell_kernel_size=conv_cell_kernel_size,
        k2m_shape=k2m_shape,
        constraints_shape=constraints_shape,
        scheduledSampleConfig=scheduledSampleConfig,
        experimentConfig=experimentConfig,
    )


def getSwinLSTMConfig(
    learning_rate: float = 1e-3,
    images_save_dir: str = "results_test/DigitalTyphoon_SwinLSTM/",
    num_save_samples: int = 10,
    experimentConfig: ExperimentConfig = None,
    input_channels: int = 1,
    input_img_size: int = 128,
    patch_size: int = 2,
    embed_dim: int = 128,
    depths_downSample: list[int] = None,
    depths_upsample: list[int] = None,
    heads_number: list[int] = None,
    window_size: int = 4,
    final_div_factor: float = 10000.0,
) -> SwinLSTMConfig:
    if experimentConfig is None:
        experimentConfig = getExperimentConfig()

    if depths_downSample is None:
        depths_downSample = [2, 6]
    if depths_upsample is None:
        depths_upsample = [6, 2]
    if heads_number is None:
        heads_number = [4, 8]

    return SwinLSTMConfig(
        learning_rate=learning_rate,
        images_save_dir=images_save_dir,
        num_save_samples=num_save_samples,
        input_channels=input_channels,
        input_img_size=input_img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths_downSample=depths_downSample,
        depths_upsample=depths_upsample,
        heads_number=heads_number,
        window_size=window_size,
        final_div_factor=final_div_factor,
        experimentConfig=experimentConfig,
    )


def getPredFormerConfig(
    learning_rate: float = 1e-3,
    images_save_dir: str = "results_test/DigitalTyphoon_PredFormer/",
    num_save_samples: int = 10,
    experimentConfig: ExperimentConfig = None,
    height: int = 224,
    width: int = 224,
    patch_size: int = 8,
    pre_seq: int = 12,
    dim: int = 256,
    num_channels: int = 1,
    heads: int = 8,
    dim_head: int = 32,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    drop_path: float = 0.0,
    scale_dim: int = 4,
    nDepth: int = 6,
    depth: int = 1,
) -> PredFormerConfig:
    if experimentConfig is None:
        experimentConfig = getExperimentConfig()

    return PredFormerConfig(
        componentConfig=PredFormerComponentConfig(
            height=height,
            width=width,
            patch_size=patch_size,
            pre_seq=pre_seq,
            dim=dim,
            num_channels=num_channels,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
            scale_dim=scale_dim,
            nDepth=nDepth,
            depth=depth,
        ),
        learning_rate=learning_rate,
        images_save_dir=images_save_dir,
        num_save_samples=num_save_samples,
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
