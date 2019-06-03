import json
import os
import argparse
from models.pseudo_labelling import NaiveKNNPseudoLabeler, SpectralPseudoLabeler, HierarchicalPseudoLabeler, \
    AggregatedPseudoLabeler
from models.embedders import FastTextEmbedder
from util.data import save_data_jsonl
import datetime


def get_args():
    args_parser = argparse.ArgumentParser()

    # Pseudo-labeling method
    args_parser.add_argument('method', type=str, choices=['nKNN', 'spectral', 'hierarchical', 'aggregated'])

    # Language to use to embed sentences
    args_parser.add_argument('--language', type=str, required=True, help='Language of data')

    # Input files, as well as file formats
    args_parser.add_argument('--input-labeled-file', type=str, required=True, help='Path to labeled file')
    args_parser.add_argument('--input-unlabeled-file', type=str, required=True, help='Path to unlabeled file')

    # Output location
    args_parser.add_argument('--output', type=str, default=f'runs/{datetime.datetime.now().isoformat()}')
    return args_parser.parse_args()


def check_args(args):
    if os.path.exists(args.output):
        raise FileExistsError(f'Path {args.output} already exists.')
    if not os.path.exists(args.input_labeled_file):
        raise FileNotFoundError(f'Path {args.input_labeled_file} does not exist.')
    if not os.path.exists(args.input_unlabeled_file):
        raise FileNotFoundError(f'Path {args.input_unlabeled_file} does not exist.')


def main():
    args = get_args()
    args_dict = vars(args)
    check_args(args)

    embedder = FastTextEmbedder(language=args.language)
    if args.method == 'nKNN':
        pseudo_labeler = NaiveKNNPseudoLabeler(embedder=embedder)
    elif args.method == 'spectral':
        pseudo_labeler = SpectralPseudoLabeler(embedder=embedder)
    elif args.method == 'hierarchical':
        pseudo_labeler = HierarchicalPseudoLabeler(embedder=embedder)
    elif args.method == 'aggregated':
        pseudo_labeler = AggregatedPseudoLabeler(embedder=embedder)
    else:
        raise NotImplementedError
    pseudo_labels = pseudo_labeler.find_pseudo_labels(
        labeled_file_path=args.input_labeled_file,
        unlabeled_file_path=args.input_unlabeled_file
    )
    os.makedirs(args.output)
    with open(os.path.join(args.output, 'args.json'), 'w') as file:
        json.dump(args_dict, file, indent=2, ensure_ascii=False)
    save_data_jsonl(pseudo_labels, os.path.join(args.output, 'pseudo_labels.jsonl'))


if __name__ == '__main__':
    main()
