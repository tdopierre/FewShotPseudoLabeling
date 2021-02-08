import os
import re
import string
from typing import List

from util.constants import FASTTEXT_MODELS_PATH, ELMO_MODELS_PATH
from util.logging import Logger
import torch

logger = Logger('FSID')


class Embedder:
    def __init__(self, language=None):
        self.language = language

    def check_download(self):
        pass

    def embed_sentences(self, sentences, **kwargs):
        pass


class FastTextEmbedder(Embedder):
    def __init__(self, language):
        import fasttext

        super(FastTextEmbedder, self).__init__(language=language)
        self.language = language  # type: str
        self.fasttext = fasttext.load_model(self.check_download())

    def check_download(self):
        current_model_path = os.path.join(FASTTEXT_MODELS_PATH, f'cc.{self.language}.300.bin')
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f'FastText model was not found at {current_model_path}.')
        return current_model_path

    def embed_sentences(self, sentences, **kwargs):
        embeddings = [self.fasttext.get_sentence_vector(str(s)) for s in sentences]
        return embeddings


class ELMoEmbedder(Embedder):
    def __init__(self, language):
        import elmoformanylangs

        super(ELMoEmbedder, self).__init__(language=language)
        self.language = language  # type:str
        self.elmo = elmoformanylangs.Embedder(self.check_download())

    def check_download(self):
        current_model_path = os.path.join(ELMO_MODELS_PATH, self.language)
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f'ELMO model was not found at {current_model_path}.')
        return current_model_path

    @staticmethod
    def _tokenize(sentence):
        sentence = re.sub(f'([{string.punctuation}])', r' \1 ', sentence.lower().strip())
        sentence_tokenized = sentence.split()
        return sentence_tokenized

    def embed_sentences(self, sentences, **kwargs):
        sentences_tokenized = [self._tokenize(s) for s in sentences]
        embedded = self.elmo.sents2elmo(sentences_tokenized, output_layer=-2)
        return [e.mean((0, 1)) for e in embedded]


class BERTEmbedder(Embedder):
    def __init__(self, config_name_or_path):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        super(BERTEmbedder, self).__init__()
        logger.info(f"Loading Encoder @ {config_name_or_path} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.bert = AutoModel.from_pretrained(config_name_or_path).to(self.device)
        logger.info(f"Encoder loaded.")

    def embed_sentences(self, sentences: List[str], detached: bool = False, **kwargs):
        batch_size = 64
        if len(sentences) > batch_size:
            vecs = [self.embed_sentences(sentences[ix:ix + batch_size], detached=detached) for ix in range(0, len(sentences), batch_size)]
            if detached:
                import numpy as np
                return np.concatenate(vecs, axis=0)
            else:
                return torch.cat(vecs, dim=0)

        padding = "max_length"

        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=padding
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        fw = self.bert.forward(**batch)
        embeddings = fw.pooler_output
        if detached:
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings
