import os
import fastText
from util.constants import FASTTEXT_MODELS_PATH
from util.logging import Logger

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
        pass

    def check_download(self):
        current_model_path = os.path.join(FASTTEXT_MODELS_PATH, f'cc.{self.language}.300.bin')
        if not os.path.exists(current_model_path):
            raise FileNotFoundError(f'FastText model was not found at {current_model_path}.')
        return current_model_path

    def embed_sentences(self, sentences):
        embeddings = [self.fasttext.get_sentence_vector(str(s)) for s in sentences]
        return embeddings
