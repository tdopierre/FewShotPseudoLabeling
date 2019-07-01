from scipy.cluster.hierarchy import linkage, to_tree
import numpy as np
from util.data import load_data_jsonl, Vocab, chunks
from util.tree import get_unique_label_trees
from util.logging import Logger
from models.embedders import Embedder
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.linalg import fractional_matrix_power, eigh

logger = Logger('FSID')


def get_nKNN_pseudo_labels(w, labeled_data, unlabeled_data, temperature=10):
    labels_vocab = Vocab([d['label'] for d in labeled_data])

    w_label = dict()

    for label in labels_vocab.labels:
        labelled_global_indices = [ix for ix, d in enumerate(labeled_data) if d['label'] == label]
        w_label[label] = w[:, labelled_global_indices]

    # Recover data
    recovered = list()
    for ix_unlabeled, data in enumerate(unlabeled_data):
        global_ix = len(labeled_data) + ix_unlabeled
        z_i = np.array([
            w_label[label][global_ix].mean()
            for label in labels_vocab.labels
        ])

        # temperature
        z_i *= temperature
        z_i_bar = np.exp(z_i)
        z_i_bar /= z_i_bar.sum()
        z_i_bar.sort()

        recovered_label = labels_vocab.labels[z_i.argmax()]
        recovered_score = z_i_bar.max()

        recovered.append(dict(
            data=data.copy(),
            pseudo_label=recovered_label,
            pseudo_label_score=float(recovered_score)
        ))
    return recovered


class NaiveKNNPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10,
            batch_size: int = None,
            **kwargs
    ):
        labeled_data = load_data_jsonl(
            labeled_file_path,
        )

        unlabeled_data = load_data_jsonl(
            unlabeled_file_path,
        )

        if not batch_size:
            batch_size = len(unlabeled_data)
        unlabeled_data_chunks = chunks(unlabeled_data, batch_size)
        n_batches = len(range(0, len(unlabeled_data), batch_size))

        all_recovered = list()
        labeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in labeled_data]))

        for batch_ix, unlabeled_data_chunk in enumerate(unlabeled_data_chunks):
            logger.info(f'Finding pseudo labels for batch {batch_ix + 1}/{n_batches}')
            unlabeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in unlabeled_data_chunk]))
            embeddings = np.concatenate((labeled_embeddings, unlabeled_embeddings), axis=0)
            w = (1 - pairwise_distances(embeddings, embeddings, metric='cosine')).astype(np.float32)
            all_recovered += get_nKNN_pseudo_labels(w, labeled_data, unlabeled_data_chunk, temperature=temperature)

        return all_recovered


class SpectralPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10,
            batch_size: int = None,
            **kwargs
    ):
        labeled_data = load_data_jsonl(
            labeled_file_path,
        )

        unlabeled_data = load_data_jsonl(
            unlabeled_file_path,
        )

        if not batch_size:
            batch_size = len(unlabeled_data)
        unlabeled_data_chunks = chunks(unlabeled_data, batch_size)
        n_batches = len(range(0, len(unlabeled_data), batch_size))

        all_recovered = list()
        labeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in labeled_data]))

        for batch_ix, unlabeled_data_chunk in enumerate(unlabeled_data_chunks):
            logger.info(f'Finding pseudo labels for batch {batch_ix + 1}/{n_batches}')
            unlabeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in unlabeled_data_chunk]))
            embeddings = np.concatenate((labeled_embeddings, unlabeled_embeddings), axis=0)

            nn = NearestNeighbors(n_neighbors=10, metric='cosine')
            nn.fit(embeddings)
            graph = nn.kneighbors_graph().toarray()
            w = (graph.T + graph > 0).astype(int)

            # D
            d = np.diag(w.sum(0))
            d_half = fractional_matrix_power(d, -0.5)

            # Normalized laplacian
            l_sym = np.eye(len(w)) - d_half @ w @ d_half

            # Eigen decomposition
            eigs = eigh(l_sym, eigvals=(1, min(31, len(l_sym) - 1)))
            normed_eigs = eigs[1] / np.sqrt(eigs[0])

            # W_prime
            w_prime = (normed_eigs @ normed_eigs.T).astype(np.float32)

            all_recovered += get_nKNN_pseudo_labels(w_prime, labeled_data, unlabeled_data_chunk,
                                                    temperature=temperature)

        return all_recovered


class HierarchicalPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10,
            batch_size: int = None,
            **kwargs
    ):
        labeled_data = load_data_jsonl(
            labeled_file_path,
        )

        unlabeled_data = load_data_jsonl(
            unlabeled_file_path,
        )
        if not batch_size:
            batch_size = len(unlabeled_data)
        unlabeled_data_chunks = chunks(unlabeled_data, batch_size)
        n_batches = len(range(0, len(unlabeled_data), batch_size))

        all_recovered = list()
        labeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in labeled_data]))

        for batch_ix, unlabeled_data_chunk in enumerate(unlabeled_data_chunks):
            logger.info(f'Finding pseudo labels for batch {batch_ix + 1}/{n_batches}')
            unlabeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in unlabeled_data_chunk]))
            labels = [d['label'] for d in labeled_data] + ['' for _ in unlabeled_data_chunk]
            labels_vocab = Vocab([d['label'] for d in labeled_data])
            embeddings = np.concatenate((labeled_embeddings, unlabeled_embeddings), axis=0)

            # Build similarity matrix
            w = (1 - pairwise_distances(embeddings, embeddings, metric='cosine')).astype(np.float32)

            # Extracts splits of W for each label. Will be used to compute score
            w_label = dict()
            for label in labels_vocab.labels:
                labelled_global_indices = [ix for ix, d in enumerate(labeled_data) if d['label'] == label]
                w_label[label] = w[:, labelled_global_indices]

            # Build hierarchical tree, bottom to top
            Z = linkage(embeddings, 'ward')
            root_tree = to_tree(Z)

            # Split tree, top to bottom
            trees = get_unique_label_trees(root_tree=root_tree, labels=labels)

            # Recover data
            recovered = list()
            for tree, path in trees:
                output = list()

                # Get all indices in the tree
                order = tree.pre_order()
                tree_labels = [labels[ix] for ix in order]

                # Case when all elements of tree are unlabelled
                if set(tree_labels) == {''}:
                    recovered += output
                    continue

                # Case when samples are mixed (labeled & unlabeled), but with a unique label
                # Get the label
                pseudo_label = [l for o, l in zip(order, tree_labels) if len(labels[o])][0]

                # Iterate over items
                for ix in order:
                    # Case if item is unlabeled
                    if labels[ix] == '':
                        # Compute score
                        global_ix = ix
                        z_i = np.array([
                            w_label[label][global_ix].mean()
                            for label in labels_vocab.labels
                        ])
                        # temperature
                        z_i *= temperature
                        z_i_bar = np.exp(z_i)
                        z_i_bar /= z_i_bar.sum()

                        pseudo_label_score = float(z_i_bar[labels_vocab(pseudo_label)])

                        # Output
                        dat = unlabeled_data_chunk[ix - len(labeled_data)].copy()
                        output.append(dict(
                            data=dat,
                            pseudo_label=pseudo_label,
                            pseudo_label_score=pseudo_label_score
                        ))
                recovered += output
            all_recovered += recovered
        return all_recovered


class AggregatedPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder
        self.nknn_labeler = NaiveKNNPseudoLabeler(embedder=self.embedder)
        self.spectral_labeler = SpectralPseudoLabeler(embedder=self.embedder)
        self.hierarchical_labeler = HierarchicalPseudoLabeler(embedder=self.embedder)

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10,
            **kwargs
    ):
        nknn_pseudo_labels = self.nknn_labeler.find_pseudo_labels(
            labeled_file_path=labeled_file_path,
            unlabeled_file_path=unlabeled_file_path,
            temperature=temperature,
            **kwargs
        )
        spectral_pseudo_labels = self.spectral_labeler.find_pseudo_labels(
            labeled_file_path=labeled_file_path,
            unlabeled_file_path=unlabeled_file_path,
            temperature=temperature,
            **kwargs
        )
        hierarchical_pseudo_labels = self.hierarchical_labeler.find_pseudo_labels(
            labeled_file_path=labeled_file_path,
            unlabeled_file_path=unlabeled_file_path,
            temperature=temperature,
            **kwargs
        )

        nknn_sentences_and_pseudo_labels = [(d['data']['input'], d['pseudo_label']) for d in nknn_pseudo_labels]
        spectral_sentences_and_pseudo_labels = [(d['data']['input'], d['pseudo_label']) for d in spectral_pseudo_labels]
        hierarchical_sentences_and_pseudo_labels = [(d['data']['input'], d['pseudo_label']) for d in
                                                    hierarchical_pseudo_labels]

        common = set(nknn_sentences_and_pseudo_labels) & set(spectral_sentences_and_pseudo_labels) & set(
            hierarchical_sentences_and_pseudo_labels)

        return [d for d in hierarchical_pseudo_labels if (d['data']['input'], d['pseudo_label']) in common]


class SelfTrainingPseudoLabeler:
    def __init__(self):
        pass

    def fit(self, path):
        pass

    def predict(self, path):
        pass


class TFIDFSelfTrainingPseudoLabeler(SelfTrainingPseudoLabeler):
    def __init__(self):
        super(TFIDFSelfTrainingPseudoLabeler, self).__init__()

    def fit(self, path):
        train_data = load_data_jsonl(path)
        self.tfidf = TfidfVectorizer()
        X = self.tfidf.fit_transform([str(d['input']) for d in train_data])
        self.logreg = LogisticRegression(C=100.0)
        self.labels_vocab = Vocab([d['label'] for d in train_data])
        y = [self.labels_vocab(d['label']) for d in train_data]
        self.logreg.fit(X, y)

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            batch_size: int = None,
            **kwargs
    ):
        self.fit(labeled_file_path)
        unlabeled_data = load_data_jsonl(unlabeled_file_path)

        if not batch_size:
            batch_size = len(unlabeled_data)
        unlabeled_data_chunks = chunks(unlabeled_data, batch_size)
        n_batches = len(range(0, len(unlabeled_data), batch_size))

        recovered = list()

        for batch_ix, unlabeled_data_chunk in enumerate(unlabeled_data_chunks):
            logger.info(f'Finding pseudo labels for batch {batch_ix + 1}/{n_batches}')
            X = self.tfidf.transform([str(d['input']) for d in unlabeled_data_chunk])
            predictions = self.logreg.predict_proba(X)
            pseudo_labels = predictions.argmax(1)
            pseudo_labels_scores = predictions.max(1)

            for original_data, pseudo_label, pseudo_label_score in zip(
                    unlabeled_data_chunk, pseudo_labels, pseudo_labels_scores):
                recovered.append(dict(
                    data=original_data.copy(),
                    pseudo_label=self.labels_vocab(pseudo_label, rev=True),
                    pseudo_label_score=float(pseudo_label_score)
                ))
        return recovered


class EmbeddedSelfTrainingPseudoLabeler(SelfTrainingPseudoLabeler):
    def __init__(self, embedder: Embedder):
        super(EmbeddedSelfTrainingPseudoLabeler, self).__init__()
        self.embedder = embedder

    def fit(self, path):
        train_data = load_data_jsonl(path)
        X = self.embedder.embed_sentences([str(d['input']) for d in train_data])
        self.logreg = LogisticRegression(C=100.0)
        self.labels_vocab = Vocab([d['label'] for d in train_data])
        y = [self.labels_vocab(d['label']) for d in train_data]
        self.logreg.fit(X, y)

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            batch_size: int = None,
            **kwargs
    ):
        self.fit(labeled_file_path)
        unlabeled_data = load_data_jsonl(unlabeled_file_path)

        if not batch_size:
            batch_size = len(unlabeled_data)
        unlabeled_data_chunks = chunks(unlabeled_data, batch_size)
        n_batches = len(range(0, len(unlabeled_data), batch_size))

        recovered = list()

        for batch_ix, unlabeled_data_chunk in enumerate(unlabeled_data_chunks):
            logger.info(f'Finding pseudo labels for batch {batch_ix + 1}/{n_batches}')
            X = self.embedder.embed_sentences([str(d['input']) for d in unlabeled_data_chunk])
            predictions = self.logreg.predict_proba(X)
            pseudo_labels = predictions.argmax(1)
            pseudo_labels_scores = predictions.max(1)

            for original_data, pseudo_label, pseudo_label_score in zip(
                    unlabeled_data_chunk, pseudo_labels, pseudo_labels_scores):
                recovered.append(dict(
                    data=original_data.copy(),
                    pseudo_label=self.labels_vocab(pseudo_label, rev=True),
                    pseudo_label_score=float(pseudo_label_score)
                ))
        return recovered
