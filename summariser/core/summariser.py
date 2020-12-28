from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration
import torch
from summariser import config
import logging

logging.basicConfig(level=config.LOGGING_LEVEL)

class Summariser:

    def __init__(self,
                 model_path: str = config.DEFAULT_MODEL_PATH,
                 use_cuda: bool = config.GPU,
                 batch_size: int = config.BATCH_SIZE):
        logging.info("Loading model...")
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.device = "cuda" \
            if torch.cuda.is_available() and self.use_cuda else "cpu"
        self.model = XLMProphetNetForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer = XLMProphetNetTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        logging.info(f"Device: {self.device}")
        logging.info(f"Num GPUs Available: {torch.cuda.device_count()}")
        logging.info(f"Model loaded")