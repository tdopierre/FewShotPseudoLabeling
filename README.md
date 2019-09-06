# Few Shot Intent Detection

## Before using the repository
### Install
This repository uses the [Pipenv](https://github.com/pypa/pipenv) environment.
```bash
# Install pipenv
pip install pipenv

# Install environment
pipenv install

# Activate environment
pipenv shell
```

### Embedding models
In order to use this repository, you must provide a path to embeddings models.
Such paths are defined in `util/constants.py`. The default path is set to `$HOME/.models/`

Default embeddings are from [FastText](https://fasttext.cc/), and can be downloaded [here](https://fasttext.cc/docs/en/crawl-vectors.html).
You must download the *bin* file, and it must be named `cc.${LANGUAGE}.300.bin`

### Input file formats
This repository uses the [JSON Lines](http://jsonlines.org/) format for input files. Example:
```text
{"input":"Switch the light on", "label":"LightOn"}
{"input":"open the door", "label":"OpenDoor"}
...
```
This is also the format of the output file containing pseudo-labels. For the unlabeled jsonl file, only the `input` key
is required 

## Usage
### Finding pseudo labels
To find pseudo labels, run the following command:
```bash
python get_pseudo_labels.py hierarchical \
    --language en \
    --input-labeled-file path/to/jsonl/labeled/file \
    --input-unlabeled-file path/to/jsonl/unlabeled/file \
    --embedder fastText \
    --output path/to/output
```
This script will compute pseudo labels using labeled and unlabeled data. The default output location is `runs/`.