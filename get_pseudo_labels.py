import argparse
from models.pseudo_labelling import NaiveKNNPseudoLabeler, SpectralPseudoLabeler, HierarchicalPseudoLabeler
from models.embedders import FastTextEmbedder


def get_args():
    args_parser = argparse.ArgumentParser()

    # Pseudo-labeling method
    args_parser.add_argument('method', type=str, choices=['nKNN', 'spectral', 'hierarchical', 'aggregated'])

    # Language to use to embed sentences
    args_parser.add_argument('--language', type=str, required=True)

    # Input files, as well as file formats
    args_parser.add_argument('--input-labeled-file', type=str, required=True)
    args_parser.add_argument('--input-unlabeled-file', type=str, required=True)

    return args_parser.parse_args()


def check_args(args):
    pass


def main():
    args = get_args()
    check_args(args)

    embedder = FastTextEmbedder(language=args.language)
    if args.method == 'nKNN':
        pseudo_labeler = NaiveKNNPseudoLabeler(embedder=embedder)
    elif args.method == 'spectral':
        pseudo_labeler = SpectralPseudoLabeler(embedder=embedder)
    elif args.method == 'hierarchical':
        pseudo_labeler = HierarchicalPseudoLabeler(embedder=embedder)
    else:
        raise NotImplementedError
    pseudo_labels = pseudo_labeler.find_pseudo_labels(
        labeled_file_path=args.input_labeled_file,
        unlabeled_file_path=args.input_unlabeled_file
    )
    print(pseudo_labels)


if __name__ == '__main__':
    main()
