from pathlib import Path
from typing import List

import torch

from libs.pyphoon2.DigitalTyphoonUtils import LOAD_DATA, SPLIT_UNIT, TRACK_COLS

from .configDatas.defaultConfigData import (
    defaultDataPathConfig,
    getBaseRNNConfig,
    getDigitalTyphoonRunConfig,
    getPhyDNetConfig,
    getPredFormerConfig,
    getSwinLSTMConfig,
)
from .ConfigType import (
    BaseRNNConfig,
    ConvLSTMConfig,
    CosineAnnealingLRConfig,
    DataModuleID,
    DataSetTimeSetting,
    DigitalTyphoonTimeSeriesDataLoadConfig,
    E3DLSTMConfig,
    ExperimentConfig,
    LightningModelID,
    MetricType,
    NetConfig,
    NetID,
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
        NetID.PhyDNet_DigitalTyphoon,
        NetID.PredFormer_DigitalTyphoon,
        NetID.E3DLSTM_DigitalTyphoon,
        NetID.SwinLSTM_DigitalTyphoon,
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    logDir = Path("./logs/")
    loggerSaveDir = "logger"
    metricsLogDir = "metrics"
    foundLrDir = "foundLr"


class NetConfigData:
    data: List[NetConfig] = [
        NetConfig(
            id=NetID.E3DLSTM_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningE3DLSTM,
            lightningModelConfig=E3DLSTMConfig(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/E3DLSTM_DigitalTyphoon",
                    kernel_size=(5, 5, 5),
                ),
                patch_size=4,
                layer_norm=False,
                window_length=2,
                window_stride=1,
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PhyDNet_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPhyDNet,
            lightningModelConfig=getPhyDNetConfig(
                img_height=128,
                img_width=128,
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.SwinLSTM_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningSwinLSTM,
            lightningModelConfig=getSwinLSTMConfig(),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredFormer_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredFormer,
            lightningModelConfig=getPredFormerConfig(
                height=128,
                width=128,
                pre_seq=12,
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.ConvLSTM_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningConvLSTM,
            lightningModelConfig=ConvLSTMConfig(
                baseRNNConfig=getBaseRNNConfig(
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
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnn_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRNN,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/PredRnn_DigitalTyphoon",
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
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnnV2_DigitalTyphoon,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnV2,
            lightningModelConfig=PredRnnV2Config(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/PredRnnV2_DigitalTyphoon",
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
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnnPP_DigitalTyphoon,
            trainEpoch=10,
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
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningConvLSTM,
            lightningModelConfig=ConvLSTMConfig(baseRNNConfig=getBaseRNNConfig()),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnn_DigitalTyphoonOne,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRNN,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/PredRnn_DigitalTyphoon"
                ),
                patch_size=4,
                useLayerNorm=False,
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnnV2_DigitalTyphoonOne,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnV2,
            lightningModelConfig=PredRnnV2Config(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/PredRnnV2_DigitalTyphoon"
                ),
                patch_size=4,
                useLayerNorm=False,
                decouple_beta=0.1,
            ),
            runConfig=getDigitalTyphoonRunConfig(),
        ),
        NetConfig(
            id=NetID.PredRnnPP_DigitalTyphoonOne,
            trainEpoch=10,
            check_val_every_n_epoch=1,
            seed=42,
            enable_progress_bar=False,
            lightningModelID=LightningModelID.LightningPredRnnPP,
            lightningModelConfig=PredRNNConfig(
                baseRNNConfig=getBaseRNNConfig(
                    images_save_dir="images/PredRnnPP_DigitalTyphoon"
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
