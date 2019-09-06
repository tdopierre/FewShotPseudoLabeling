import os
import fastText
import elmoformanylangs
import re
import string
from util.constants import FASTTEXT_MODELS_PATH, ELMO_MODELS_PATH
from util.logging import Logger
import numpy as np

logger = Logger('FSID')


class Embedder:
    def __init__(self, language):
        self.language = language

    def check_download(self):
        pass

    def embed_sentences(self, sentences):
        pass


class FastTextEmbedder(Embedder):
    def __init__(self, language):
        super(FastTextEmbedder, self).__init__(language=language)
        self.language = language  # type: str
        self.fasttext = fastText.load_model(self.check_download())  # type: fastText.FastText

    def check_download(self):
        current_model_path = os.path.join(FASTTEXT_MODELS_PATH, f'cc.{self.language}.300.bin')
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f'FastText model was not found at {current_model_path}.')
        return current_model_path

    def embed_sentences(self, sentences):
        embeddings = [self.fasttext.get_sentence_vector(str(s)) for s in sentences]
        return embeddings


class ELMoEmbedder(Embedder):
    def __init__(self, language):
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

    def embed_sentences(self, sentences):
        sentences_tokenized = [self._tokenize(s) for s in sentences]
        embedded = self.elmo.sents2elmo(sentences_tokenized, output_layer=-2)
        return [e.mean((0, 1)) for e in embedded]
