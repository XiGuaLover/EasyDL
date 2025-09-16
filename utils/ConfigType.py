from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch.nn
from libs.pyphoon2.DigitalTyphoonUtils import LOAD_DATA, SPLIT_UNIT
from torch.optim.optimizer import Optimizer


class NetID(Enum):
    PredRnn_DigitalTyphoon = "PredRnn_DigitalTyphoon"
    PredRnnV2_DigitalTyphoon = "PredRnnV2_DigitalTyphoon"
    PredRnnPP_DigitalTyphoon = "PredRnnPP_DigitalTyphoon"
    ConvLSTM_DigitalTyphoon = "ConvLSTM_DigitalTyphoon"
    PredRnn_DigitalTyphoonOne = "PredRnn_DigitalTyphoonOne"
    PredRnnV2_DigitalTyphoonOne = "PredRnnV2_DigitalTyphoonOne"
    PredRnnPP_DigitalTyphoonOne = "PredRnnPP_DigitalTyphoonOne"
    ConvLSTM_DigitalTyphoonOne = "ConvLSTM_DigitalTyphoonOne"


class LightningModelID(Enum):
    LightningPredRnnPP = "LightningPredRnnPP"
    LightningPredRNN = "LightningPredRNN"
    LightningConvLSTM = "LightningConvLSTM"
    LightningPredRnnV2 = "LightningPredRnnV2"


class MetricType(Enum):
    MSE = "MSE"
    MAE = "MAE"
    RMSE = "RMSE"
    SSIM = "SSIM"


@dataclass
class OneCycleLRConfig:
    pct_start: float = 0.3
    div_factor: float = 25
    final_div_factor: float = 10000


@dataclass
class CosineAnnealingLRConfig:
    T_max: int  # Maximum number of iterations
    eta_min: float = 0  # Minimum learning rate


class DataModuleID(Enum):
    DigitalTyphoon = "DigitalTyphoon"


@dataclass
class DataSetTimeSetting:
    trainTime: tuple
    valTime: tuple
    testTime: tuple


@dataclass
class DigitalTyphoonDataPathConfig:
    dataDir: str
    metadataJsonPath: str
    metadataDir: str
    imageDir: str


@dataclass
class DigitalTyphoonBaseDataLoadConfig:
    batchSize: int
    labels: List[str]
    standardizeRange: Tuple[int, int]
    downSampleSize: Tuple[int, int]
    numWorkers: int
    pathConfig: DigitalTyphoonDataPathConfig


@dataclass
class DigitalTyphoonDataLoadConfig(DigitalTyphoonBaseDataLoadConfig):
    splitBy: SPLIT_UNIT
    loadData: LOAD_DATA
    cropped: bool
    loadImgsIntoMemory: bool = False
    pin_memory: bool = False
    datasetSplitByRatio: Tuple[float, float, float] | None = None
    datasetSplitByTime: DataSetTimeSetting | None = None


@dataclass
class DigitalTyphoonTimeSeriesDataLoadConfig(DigitalTyphoonDataLoadConfig):
    inputSeqLen: int = 1
    targetSeqLen: int = 1
    returnLabels: bool = False


@dataclass
class ScheduledSampleConfig:
    initial_sampling_rate: float
    enable_scheduled_sampling: bool
    stop_sampling_training_global_step: int
    sampling_decay_rate: float
    patch_size: int


@dataclass
class ExperimentConfig:
    metrics: List[MetricType] = field(default_factory=lambda: [MetricType.MSE])
    optimizer: Optimizer = torch.optim.Adam
    criterionFunction: torch.nn.Module = torch.nn.MSELoss
    schedulerConfig: Optional[OneCycleLRConfig] = None


@dataclass
class BaseConfig:
    learning_rate: float
    img_channel: int
    img_width: int
    img_height: int
    num_save_samples: int
    images_save_dir: str


@dataclass
class BaseRNNConfig(BaseConfig):
    kernel_size: Tuple[int, int]

    num_hidden: List[int]
    stride: int
    scheduledSampleConfig: ScheduledSampleConfig

    experimentConfig: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass
class PredRNNConfig:
    baseRNNConfig: BaseRNNConfig
    patch_size: int
    useLayerNorm: bool


@dataclass
class PredRnnV2Config(PredRNNConfig):
    decouple_beta: float


@dataclass
class ConvLSTMConfig:
    baseRNNConfig: BaseRNNConfig


@dataclass
class RunConfig:
    # Dataset parameters
    dataloadConfig: Union[DigitalTyphoonDataLoadConfig,]

    findLr: bool = False
    saveCheckpoint: bool = False
    resumeCheckpointFilePath: Optional[str] = None
    resumeTrainFromCheckpoint: bool = False
    earlyStop: bool = True
    test: bool = False
    testCheckpointFilePath: Optional[str] = None


@dataclass
class NetConfig:
    id: NetID
    lightningModelID: LightningModelID
    runConfig: RunConfig
    lightningModelConfig: Union[
        PredRNNConfig,
        ConvLSTMConfig,
    ]
    specifyDevices: List[int] = field(default_factory=lambda: [0])
    trainEpoch: int = 10
    check_val_every_n_epoch: int = 1
    seed: Optional[int] = None
    enable_progress_bar: bool = True
    profiler: Optional[str] = None
