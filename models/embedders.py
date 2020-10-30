import os
import fastText
import elmoformanylangs
import re
import string
from util.constants import FASTTEXT_MODELS_PATH, ELMO_MODELS_PATH
from util.logging import Logger
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer, BertConfig, AutoConfig, AutoModel, AutoTokenizer
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


class BERTEmbedder(nn.Module):
    def __init__(self, config_name_or_path, device=torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")):
        super(BERTEmbedder, self).__init__()
        self.device = device
        logger.info(f"Loading Encoder @ {config_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.bert = AutoModel.from_pretrained(config_name_or_path).to(self.device)
        logger.info(f"Encoder loaded.")

    def forward(self, sentences):
        batch_size = 2
        if len(sentences) > batch_size:
            return torch.cat([self.forward(sentences[i:i + batch_size]) for i in range(0, len(sentences), batch_size)], 0)
        encoded_plus = [self.tokenizer.encode_plus(s, max_length=256) for s in sentences]
        max_len = max([len(e['input_ids']) for e in encoded_plus])

        input_ids = list()
        attention_masks = list()
        token_type_ids = list()

        for e in encoded_plus:
            e['input_ids'] = e['input_ids'][:max_len]
            e['token_type_ids'] = e['token_type_ids'][:max_len]
            pad_len = max_len - len(e['input_ids'])
            input_ids.append(e['input_ids'] + (pad_len) * [self.tokenizer.pad_token_id])
            attention_masks.append([1 for _ in e['input_ids']] + [0] * pad_len)
            token_type_ids.append(e['token_type_ids'] + [0] * pad_len)

        _, x = self.bert.forward(input_ids=torch.Tensor(input_ids).long().to(self.device),
                                 attention_mask=torch.Tensor(attention_masks).long().to(self.device),
                                 token_type_ids=torch.Tensor(token_type_ids).long().to(self.device))

        return x.cpu().detach().numpy().tolist()
