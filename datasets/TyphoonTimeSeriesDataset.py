from torch.utils.data import Dataset
from libs.pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import numpy as np
from typing import Tuple, List


class TyphoonTimeSeriesDataset(Dataset):
    r"""creating time-series sequences of typhoon images"""

    def __init__(
        self,
        digitalTyphoonDataset: DigitalTyphoonDataset,
        typhoonIndices: List[int],
        seqLength: int,
    ):
        # Starting indices of valid image sequences， of all typhoon sequences
        self.sequenceStartIndexes = []
        self.seqLength = seqLength
        self.digitalTyphoonDataset = digitalTyphoonDataset
        self.typhoonIndices = typhoonIndices
        for index in typhoonIndices:
            typhoon = digitalTyphoonDataset.get_ith_sequence(index)
            typhoonImagesNum = typhoon.get_num_images()
            indexStartOfTyphoonImageAtDigitalTyphoonDataset = (
                digitalTyphoonDataset._seq_str_to_first_total_idx[typhoon.sequence_str]
            )
            sequenceNum = typhoonImagesNum - seqLength
            # todo
            if sequenceNum > 0:
                for i in range(sequenceNum):
                    self.sequenceStartIndexes.append(
                        indexStartOfTyphoonImageAtDigitalTyphoonDataset + i
                    )

        print("TyphoonTimeSeriesDataset initialized.")

    def __len__(self):
        return len(self.sequenceStartIndexes)

    def __getitem__(self, idx) -> Tuple[List[np.ndarray], np.number | np.ndarray]:
        r"""Retrieve a sequence of images and the label for the next image."""
        imageIndex = self.sequenceStartIndexes[idx]
        self.digitalTyphoonDataset.get_images_by_sequence = False

        imagesSequence: List[np.ndarray] = [
            np.expand_dims(self.digitalTyphoonDataset[i][0], axis=0)
            for i in range(imageIndex, imageIndex + self.seqLength)
        ]
        nextImageLabels: np.number | np.ndarray = self.digitalTyphoonDataset[
            imageIndex + self.seqLength
        ][1]

        return (imagesSequence, nextImageLabels)
