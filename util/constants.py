import os

# Models path is, by default, ${HOME}/.models
MODELS_PATH = os.path.join(os.getenv('HOME'), '.models')
FASTTEXT_MODELS_PATH = os.path.join(MODELS_PATH, 'fasttext')
ELMO_MODELS_PATH = os.path.join(MODELS_PATH, 'elmo')
