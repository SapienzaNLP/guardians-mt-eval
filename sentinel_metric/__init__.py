import logging

from .models import load_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
