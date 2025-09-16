# from torchvision.transforms.functional import center_crop
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from libs.pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from torch import nn
from torch.utils.data import DataLoader
from utils.ConfigType import (
    DataSetTimeSetting,
    DigitalTyphoonDataPathConfig,
    DigitalTyphoonTimeSeriesDataLoadConfig,
)

from .SequenceTyphoonDataset import SequenceTyphoonDataset
from .TyphoonTimeSeriesDataset import TyphoonTimeSeriesDataset
from .TyphoonTimeSeriesFilterDataset import TyphoonTimeSeriesFilterDataset


class TyphoonTimeSeriesModule(pl.LightningDataModule):
    """Typhoon Dataset Module using lightning architecture"""

    def __init__(
        self,
        dataPathConfig: DigitalTyphoonDataPathConfig,
        batch_size,
        num_workers,
        labels,
        load_data,
        datasetSplitByRatio,
        standardize_range,
        downSample_size,
        cropped,
        datasetSplitByTime: DataSetTimeSetting,
        inputSeqLen: int,
        targetSeqLen: int,
        loadImgsIntoMemory: bool,
        pin_memory: bool,
        returnLabels: bool = False,
        corruption_ceiling_pct=100,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataPathConfig = dataPathConfig
        self.load_data = load_data
        self.labels = labels

        self.datasetSplitByRatio = datasetSplitByRatio
        self.datasetSplitByTime = datasetSplitByTime
        self.standardize_range = standardize_range
        self.downSample_size = downSample_size
        self.cropped = cropped

        self.inputSeqLen = inputSeqLen
        self.targetSeqLen = targetSeqLen
        self.loadImgsIntoMemory = loadImgsIntoMemory
        self.pin_memory = pin_memory
        self.returnLabels = returnLabels

        self.corruption_ceiling_pct = corruption_ceiling_pct

    def setup(self, stage):
        print("Setup Dataloader ...")

        if self.datasetSplitByRatio is not None:
            print("Split dataset by ratio")
            # Load Dataset
            filterDataset = TyphoonTimeSeriesFilterDataset(
                seqLength=self.inputSeqLen,
                image_dir=self.dataPathConfig.imageDir,
                metadata_dir=self.dataPathConfig.metadataDir,
                metadata_json=self.dataPathConfig.metadataJsonPath,
                labels=self.labels,
                load_data_into_memory=self.load_data,
                filter_func=self.image_filter,
                transform_func=self.transform_func,
                spectrum="Infrared",
                verbose=False,
                returnLabels=self.returnLabels,
            )

            self.train_set, self.val_set, self.test_set = (
                filterDataset._random_split_by_sequence(self.datasetSplitByRatio)
            )

            self.train_set = TyphoonTimeSeriesDataset(
                filterDataset, self.train_set.indices, filterDataset.seqLength
            )
            self.val_set = TyphoonTimeSeriesDataset(
                filterDataset, self.val_set.indices, filterDataset.seqLength
            )
            self.test_set = TyphoonTimeSeriesDataset(
                filterDataset, self.test_set.indices, filterDataset.seqLength
            )

        else:
            print("self.datasetSplitByTime", self.datasetSplitByTime)
            trainFilterDataset = SequenceTyphoonDataset(
                seqLength=self.inputSeqLen,
                image_dir=self.dataPathConfig.imageDir,
                metadata_dir=self.dataPathConfig.metadataDir,
                metadata_json=self.dataPathConfig.metadataJsonPath,
                labels=self.labels,
                load_data_into_memory=self.load_data,
                inputSeqLen=self.inputSeqLen,
                targetSeqLen=self.targetSeqLen,
                loadImgsIntoMemory=self.loadImgsIntoMemory,
                # filter_func=self.imageFilterWithTime,
                sequenceFilterWhenPopImageToSequences=self.sequenceFilterWithTime,
                transform_func=self.transform_func,
                spectrum="Infrared",
                verbose=False,
                returnLabels=self.returnLabels,
            )

            self.train_set = trainFilterDataset.getSubsetByDate(
                self.datasetSplitByTime.trainTime
            )
            self.val_set = trainFilterDataset.getSubsetByDate(
                self.datasetSplitByTime.valTime
            )
            self.test_set = trainFilterDataset.getSubsetByDate(
                self.datasetSplitByTime.testTime
            )

            print(len(trainFilterDataset.sequences))

            aaa = DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )

            print((len(aaa.dataset)), " ``````````````````````")
            bbb = DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )

            print((len(bbb.dataset)), " ``````````````````````")

            ccc = DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
            print((len(ccc.dataset)), " ``````````````````````")

        print("TyphoonTimeSeriesModule setup done.")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def sequenceFilterWithTime(self, sequenceID: str):
        year: int = int(sequenceID[:4])

        inTrainTime: bool = (
            year >= self.datasetSplitByTime.trainTime[0]
            and year <= self.datasetSplitByTime.trainTime[1]
        )
        inTestTime: bool = (
            year >= self.datasetSplitByTime.testTime[0]
            and year <= self.datasetSplitByTime.testTime[1]
        )
        inValTime: bool = (
            year >= self.datasetSplitByTime.valTime[0]
            and year <= self.datasetSplitByTime.valTime[1]
        )

        return inTrainTime or inTestTime or inValTime

    def image_filter(self, image: DigitalTyphoonImage) -> bool:
        return (
            (image.grade() < 7)
            # and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )

    def imageFilterWithTime(self, image: DigitalTyphoonImage):
        inTrainTime: bool = (
            image.year() >= self.datasetSplitByTime.trainTime[0]
            and image.year() <= self.datasetSplitByTime.trainTime[1]
        )
        inTestTime: bool = (
            image.year() >= self.datasetSplitByTime.testTime[0]
            and image.year() <= self.datasetSplitByTime.testTime[1]
        )
        inValTime: bool = (
            image.year() >= self.datasetSplitByTime.valTime[0]
            and image.year() <= self.datasetSplitByTime.valTime[1]
        )

        return self.image_filter(image) and (inTrainTime or inTestTime or inValTime)

    def trainImageFilterWithTime(self, image: DigitalTyphoonImage, seq_id: str) -> bool:
        result = (
            self.image_filter(image)
            and (image.year() >= self.datasetSplitByTime.trainTime[0])
            and (image.year() <= self.datasetSplitByTime.trainTime[1])
        )
        # print("result", result)
        return result

    def testImageFilterWithTime(self, image: DigitalTyphoonImage, seq_id: str):
        return (
            self.image_filter(image)
            and (image.year() >= self.datasetSplitByTime.testTime[0])
            and (image.year() <= self.datasetSplitByTime.testTime[1])
        )

    def valImageFilterWithTime(self, image: DigitalTyphoonImage, seq_id: str):
        return (
            self.image_filter(image)
            and (image.year() >= self.datasetSplitByTime.valTime[0])
            and (image.year() <= self.datasetSplitByTime.valTime[1])
        )

    def transform_func(self, image_batch):
        """transform function applied on the images for pre-processing"""
        image_batch = np.clip(
            image_batch, self.standardize_range[0], self.standardize_range[1]
        )
        image_batch = (image_batch - self.standardize_range[0]) / (
            self.standardize_range[1] - self.standardize_range[0]
        )
        if self.downSample_size != (512, 512):
            image_batch = torch.Tensor(image_batch)
            if self.cropped:
                image_batch = torch.reshape(
                    image_batch, [1, 1, image_batch.size()[0], image_batch.size()[1]]
                )
                image_batch = nn.functional.interpolate(
                    image_batch,
                    size=self.downSample_size,
                    mode="bilinear",
                    align_corners=False,
                )
                image_batch = torch.reshape(
                    image_batch, [image_batch.size()[2], image_batch.size()[3]]
                )
                image_batch = image_batch.numpy()
        return image_batch


if __name__ == "__main__":
    cfg = DigitalTyphoonTimeSeriesDataLoadConfig(
        batchSize=16,
        labels=["wind"],
        splitBy="sequence",
        loadData="all_data",
        # datasetSplitByRatio=(0.8, 0.2, 0),
        datasetSplitByTime=DataSetTimeSetting(
            # trainTime=(1978, 2014),
            trainTime=(1978, 1981),
            valTime=(2015, 2016),
            testTime=(2023, 2024),
        ),
        standardizeRange=(170, 350),
        downSampleSize=(224, 224),
        cropped=False,
        numWorkers=32,
        pathConfig=DigitalTyphoonDataPathConfig(
            dataDir=Path("/root/disk/datasets/WP"),
            metadataJsonPath=Path("/root/disk/datasets/WP/metadata.json"),
            metadataDir=Path("/root/disk/datasets/WP/metadata/"),
            imageDir=Path("/root/disk/datasets/WP/image/"),
        ),
    )

    dataModule = TyphoonTimeSeriesModule(
        dataPathConfig=cfg.pathConfig,
        batch_size=cfg.batchSize,
        # num_workers=cfg.runConfig.numWorkers,
        labels=cfg.labels,
        # no split_by=cfg.splitBy.name.lower(),
        num_workers=cfg.numWorkers,
        load_data=cfg.loadData,
        datasetSplitByRatio=cfg.datasetSplitByRatio,
        datasetSplitByTime=cfg.datasetSplitByTime,
        standardize_range=cfg.standardizeRange,
        downSample_size=cfg.downSampleSize,
        cropped=cfg.cropped,
    )

    dataModule.setup()
