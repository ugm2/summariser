from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration
import torch
from summariser import config
import logging
from itertools import chain
from tqdm import tqdm
from typing import List

logging.basicConfig(level=config.LOGGING_LEVEL)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

class Summariser:

    def __init__(self,
                 model_path: str = config.DEFAULT_MODEL_PATH,
                 use_cuda: bool = config.GPU,
                 batch_size: int = config.BATCH_SIZE):
        '''
        Constructs all the necessary attributes for the Summariser object.

        Parameters
        ----------
            model_path : str
                path to the summarisation model
            use_cuda : bool
                whether to use CUDA or not (if available)
            batch_size : int
                number of samples passed to the model to predict at once

        '''
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

    def summarise(self, sentences: List[str]):
        '''
        Generate summaries from input sentences

        Parameters
        ----------
            sentences : list of str
                list of sentences. Each sentence gets its own summary
        '''
        # Break list into chunks of size self.batch_size
        sentences_chunks = list(chunks(sentences, self.batch_size))

        total_summaries = []
        # Loop over batches of sentences
        for sentences_batch in tqdm(sentences_chunks):
            # Tokenize batch of sentences
            inputs = self.tokenizer(sentences_batch, padding=True, max_length=256, return_tensors='pt')
            # Send inputs to device
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            # Generate summary IDs
            summary_ids = self.model.generate(inputs['input_ids'],
                                              num_beams=4, max_length=100,
                                              early_stopping=True)
            # Decode summary IDs into string sentences
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            total_summaries += summaries
            # Clear cuda cache if needed
            torch.cuda.empty_cache()

        return total_summaries