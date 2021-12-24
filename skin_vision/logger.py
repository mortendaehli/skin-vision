import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        RotatingFileHandler('.log', maxBytes=2000, backupCount=10),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger("skin_vision")
