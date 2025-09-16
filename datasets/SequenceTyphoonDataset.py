import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Union

import h5py
import torch
from libs.pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from libs.pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage
from libs.pyphoon2.DigitalTyphoonSequence import DigitalTyphoonSequence
from torch.utils.data import Subset


@dataclass
class SequenceImageIndex:
    sequenceID: str
    firstImageIndex: int


class SequenceTyphoonDataset(DigitalTyphoonDataset):
    def __init__(
        self,
        seqLength,
        image_dir: str,
        metadata_dir: str,
        metadata_json: str,
        labels,
        inputSeqLen: int,
        targetSeqLen: int,
        loadImgsIntoMemory: bool,
        spectrum="Infrared",
        load_data_into_memory=False,
        ignore_list=None,
        filter_func=None,
        sequenceFilterWhenPopImageToSequences=None,
        transform_func=None,
        transform=None,
        verbose=True,
        returnLabels: bool = False,
    ) -> None:
        print(
            "SequenceTyphoonDataset init ... time:",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        if os.path.isdir(image_dir) and not str(image_dir).endswith("/"):
            image_dir = str(image_dir) + "/"

        if os.path.isdir(metadata_dir) and not str(metadata_dir).endswith("/"):
            metadata_dir = str(metadata_dir) + "/"

        super().__init__(
            image_dir=image_dir,
            metadata_dir=metadata_dir,
            metadata_json=metadata_json,
            labels=labels,
            split_dataset_by="sequence",
            spectrum=spectrum,
            get_images_by_sequence=True,
            load_data_into_memory=load_data_into_memory,
            ignore_list=ignore_list,
            filter_func=filter_func,
            sequenceFilterWhenPopImageToSequences=sequenceFilterWhenPopImageToSequences,
            transform_func=transform_func,
            transform=transform,
            verbose=verbose,
        )

        self.labels = labels
        self.returnLabels = returnLabels
        self.inputSeqLen = inputSeqLen
        self.targetSeqLen = targetSeqLen
        self.sequenceImageIndexList: List[SequenceImageIndex] = []
        totalLength = self.inputSeqLen + self.targetSeqLen
        self._seq_cache = {}  # cached seq

        # Post process sequences filter out too short sequences
        def filter_sequences(sequence: DigitalTyphoonSequence):
            if sequence.get_num_images() < totalLength:
                return True
            return False

        count = 0

        for seq in self.sequences:
            if seq.get_num_images() > 0:
                if filter_sequences(seq):
                    self.number_of_images -= seq.get_num_images()
                    seq.images.clear()
                    self.number_of_nonempty_sequences -= 1
                    count += 1

                else:
                    seq_id = seq.sequence_str
                    self._seq_cache[seq_id] = seq
                    seqNums = seq.get_num_images() - totalLength + 1
                    for i in range(seqNums):
                        self.sequenceImageIndexList.append(
                            SequenceImageIndex(sequenceID=seq_id, firstImageIndex=i)
                        )

                    if loadImgsIntoMemory:
                        for img in seq.images:
                            img.load_imgs_into_mem = loadImgsIntoMemory
                            img.image()

        print(
            "SequenceTyphoonDataset init finish ... time:",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    def __getitem__(
        self, idx
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        sequenceImageIndex: SequenceImageIndex = self.sequenceImageIndexList[idx]
        seq_id = sequenceImageIndex.sequenceID
        seq = self._seq_cache[seq_id]

        inputImageIndex = sequenceImageIndex.firstImageIndex
        targetImageIndex = inputImageIndex + self.inputSeqLen
        targetImageEndIndex = targetImageIndex + self.targetSeqLen
        images: List[torch.Tensor] = [
            torch.from_numpy(image.image()).float().unsqueeze(0)  # add channel 1 to dim
            for image in seq.images[inputImageIndex:targetImageIndex]
        ]
        targetImages: List[torch.Tensor] = [
            torch.from_numpy(image.image()).float().unsqueeze(0)
            for image in seq.images[targetImageIndex:targetImageEndIndex]
        ]
        if not self.returnLabels:
            return (torch.stack(images), torch.stack(targetImages))

        inputLabels = [
            torch.tensor(image.value_from_string(self.labels[0]))
            for image in seq.images[inputImageIndex:targetImageIndex]
        ]
        targetLabels = [
            torch.tensor(image.value_from_string(self.labels[0]))
            for image in seq.images[targetImageIndex:targetImageEndIndex]
        ]
        return (
            torch.stack(images),
            torch.stack(targetImages),
            torch.stack(inputLabels),
            torch.stack(targetLabels),
        )

    def getSequenceSubsetByDate(
        self,
        date: Tuple[int, int],
    ):
        non_empty_sequence_indices: List[int] = []
        for idx in range(len(self.sequences)):
            seq = self.sequences[idx]
            if seq.get_num_images() <= 0:
                continue

            year: int = int(seq.get_sequence_str()[:4])
            if year >= date[0] and year <= date[1]:
                non_empty_sequence_indices.append(idx)

        return Subset(self, non_empty_sequence_indices)

    def getSubsetByDate(
        self,
        date: Tuple[int, int],
    ):
        indices: List[int] = []
        for index, sequenceImageIndex in enumerate(self.sequenceImageIndexList):
            year: int = int(sequenceImageIndex.sequenceID[:4])
            if year >= date[0] and year <= date[1]:
                indices.append(index)
        return Subset(self, indices)

    def getH5ImageAsTensor(self, typhoonImage: DigitalTyphoonImage) -> torch.Tensor:
        with h5py.File(typhoonImage.image_filepath, "r") as h5f:
            image = torch.tensor(h5f.get(typhoonImage.spectrum), dtype=torch.float)
        return image
