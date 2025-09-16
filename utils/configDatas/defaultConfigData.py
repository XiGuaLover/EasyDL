from pathlib import Path
from typing import Dict

from ..ConfigType import (
    DataModuleID,
    DigitalTyphoonDataPathConfig,
)

defaultDataPathConfig: Dict[DataModuleID, any] = {
    DataModuleID.DigitalTyphoon: DigitalTyphoonDataPathConfig(
        dataDir=Path("/root/disk/datasets/WP"),
        metadataJsonPath=Path("/root/disk/datasets/WP/metadata.json"),
        metadataDir=Path("/root/disk/datasets/WP/metadata/"),
        imageDir=Path("/root/disk/datasets/WP/image/"),
    )
}
