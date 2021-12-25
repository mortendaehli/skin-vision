from pathlib import Path
from pydantic import BaseSettings

import skin_vision
from typing import List


class Config(BaseSettings):

    ROOT_DATA_PATH: Path = Path(skin_vision.__file__).parent.parent / ".tmp"
    ROOT_TMP_PATH: Path = Path(skin_vision.__file__).parent.parent / ".tmp"
    MODEL_WEIGHTS_PATH: Path = Path(skin_vision.__file__).parent.parent / "model_weights"

    BATCH_SIZE: int = 64  # from 128
    EPOCHS: int = 2  # from 50
    IMAGE_SIZE: List[int] = [1024, 1024]

    # LEARNING RATE
    LR_START: float = 0.000003
    LR_EXP_DECAY: float = 0.96
    LR_DECAY_STEPS: int = 20


config = Config()
