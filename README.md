# Few-shot Pseudo-Labeling for Intent Detection

This repository contains code for the paper [Few-shot Pseudo-Labeling for Intent Detection](https://www.aclweb.org/anthology/2020.coling-main.438/)
## Before using the repository
### Install
This repository uses the [virtualenv](https://virtualenv.pypa.io/en/latest/) environment.
```bash
# Install pipenv
python3 -m virtualenv .venv --python=python3.6

# Install environment
.venv/bin/pip install -r requirements.txt

# Activate environment
source .venv/bin/activate
```

### Embedding models
In order to use this repository, you must provide a path to embeddings models.
Such paths are defined in `util/constants.py`. The default path is set to `$HOME/.models/`

You can also use transformers, by specifying either a name of a model or a path to it.

### Input file formats
This repository uses the [JSON Lines](http://jsonlines.org/) format for input files. Example:
```text
{"sentence":"Switch the light on", "label":"LightOn"}
{"sentence":"open the door", "label":"OpenDoor"}
...
```
This is also the format of the output file containing pseudo-labels. For the unlabeled jsonl file, only the `input` key
is required 

## Usage
### Finding pseudo labels
To find pseudo labels, run the following command:
```bash
python get_pseudo_labels.py fold-unfold \
    --embedder bert \
    --model-name-or-path bert-base-cased \
    --input-labeled-file data/datasets/snips/few-shot_final/01/n-samples-005/support.jsonl \
    --input-unlabeled-file data/datasets/snips/few-shot_final/01/n-samples-005/query.jsonl \
    -v
```

This script will compute pseudo labels using labeled and unlabeled data. The default output location is `runs/`.

## Reference
If you use the data or codes in this repository, please cite our paper.
```bash
@inproceedings{dopierre-etal-2020-shot,
    title = "Few-shot Pseudo-Labeling for Intent Detection",
    author = "Dopierre, Thomas  and Gravier, Christophe  and Subercaze, Julien  and Logerais, Wilfried",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    year = "2020",
}
```