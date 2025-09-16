import os

from libs.pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from libs.pyphoon2.DigitalTyphoonImage import DigitalTyphoonImage


class TyphoonTimeSeriesFilterDataset(DigitalTyphoonDataset):
    r"""split_dataset_by="sequence"
    return all sequences that have at least seqLength+1 images
    """

    def __init__(
        self,
        seqLength,
        image_dir: str,
        metadata_dir: str,
        metadata_json: str,
        labels,
        spectrum="Infrared",
        load_data_into_memory=False,
        ignore_list=None,
        filter_func=None,
        transform_func=None,
        transform=None,
        verbose=False,
        returnLabels: bool = False,
    ):
        self.seqLength = seqLength

        # to adapt the digital typhoon dataset setting
        # as usual, we don't need this process
        if os.path.isdir(image_dir) and not str(image_dir).endswith("/"):
            image_dir = str(image_dir) + "/"

        if os.path.isdir(metadata_dir) and not str(metadata_dir).endswith("/"):
            metadata_dir = str(metadata_dir) + "/"

        self._imageInitFilterFunc = filter_func

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
            filter_func=self.imageFilter,
            transform_func=transform_func,
            transform=transform,
            verbose=verbose,
        )

        assert not returnLabels, (
            "returnLabels must be False, returnLabels is not supported"
        )

        print("TyphoonTimeSeriesFilterDataset initialized.")

    def imageFilter(self, image: DigitalTyphoonImage) -> bool:
        sequence = self._get_seq_from_seq_str(image.sequence_id())
        isHaveEnoughImages = sequence.num_original_images > self.seqLength

        return isHaveEnoughImages and (
            self._imageInitFilterFunc is None or self._imageInitFilterFunc(image)
        )

    def numOfValidTyphoons(self):
        return self.number_of_nonempty_sequences

    def getAllTyphoonsNumber(self):
        return len(self.sequences)
