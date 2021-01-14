# Few-shot Pseudo-Labeling for Intent Detection

This repository contains code for the paper **Few-shot Pseudo-Labeling for Intent Detection**.
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

You can also use transformers, by specifying either a name of a model or a path to it.

[FastText](https://fasttext.cc/) embeddings can be downloaded [here](https://fasttext.cc/docs/en/crawl-vectors.html).
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