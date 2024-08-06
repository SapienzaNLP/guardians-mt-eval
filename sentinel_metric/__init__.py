import logging

from .models import load_from_checkpoint, download_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
