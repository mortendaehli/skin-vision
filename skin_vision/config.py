from pathlib import Path
from pydantic import BaseSettings

import skin_vision
from typing import Tuple


class Config(BaseSettings):

    ROOT_DATA_PATH: Path = Path(skin_vision.__file__).parent.parent / ".tmp"
    ROOT_TMP_PATH: Path = Path(skin_vision.__file__).parent.parent / ".tmp"
    MODEL_WEIGHTS_PATH: Path = Path(skin_vision.__file__).parent.parent / "model_weights"
    BUFFER_SIZE: int = 584
    BATCH_SIZE: int = 32  # from 128
    EPOCHS: int = 50  # from 50
    IMAGE_SIZE: Tuple[int, int] = (256, 256)
    SHAPE: Tuple[int, int] = (256, 256)


config = Config()
