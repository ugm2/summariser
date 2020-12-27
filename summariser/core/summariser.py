from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration
import torch
from summariser.core import config
import logging

logger = logging.basicConfig(level=logging.INFO)

class Summariser:

    def __init__(self,
                 model_path: str = config.DEFAULT_MODEL_PATH,
                 use_cuda: bool = config.GPU,
                 batch_size: int = config.BATCH_SIZE):
        logger.info("Loading model...")
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.device = "cuda" \
            if torch.cuda.is_available() and self.use_cuda else "cpu"
        self.tokenizer = XLMProphetNetTokenizer(self.model_path)
        self.model = XLMProphetNetForConditionalGeneration(self.model_path)
        self.model.to(self.device)
        logger.info("Device: ", self.device)
        logger.info("Num GPUs Available: ", torch.cuda.device_count())
        logger.info("Model loaded")