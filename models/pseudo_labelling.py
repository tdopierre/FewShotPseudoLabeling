import numpy as np
from util.data import load_data_jsonl, Vocab
from models.embedders import Embedder
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.linalg import fractional_matrix_power, eigh


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
        recovered_score = z_i.max()

        recovered.append(dict(
            input=data['input'],
            pseudo_label=recovered_label,
            pseudo_label_score=recovered_score
        ))
    return recovered


class NaiveKNNPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10
    ):
        labeled_data = load_data_jsonl(
            labeled_file_path,
        )

        unlabeled_data = load_data_jsonl(
            unlabeled_file_path,
        )

        labeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in labeled_data]))
        unlabeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in unlabeled_data]))

        embeddings = np.concatenate((labeled_embeddings, unlabeled_embeddings), axis=0)

        w = (1 - pairwise_distances(embeddings, embeddings, metric='cosine')).astype(np.float32)

        return get_nKNN_pseudo_labels(w, labeled_data, unlabeled_data, temperature=temperature)


class SpectralPseudoLabeler:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder  # type: Embedder

    def find_pseudo_labels(
            self,
            labeled_file_path: str,
            unlabeled_file_path: str,
            temperature: int = 10
    ):
        labeled_data = load_data_jsonl(
            labeled_file_path,
        )

        unlabeled_data = load_data_jsonl(
            unlabeled_file_path,
        )

        labeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in labeled_data]))
        unlabeled_embeddings = np.array(self.embedder.embed_sentences([d['input'] for d in unlabeled_data]))

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
        eigs = eigh(l_sym, eigvals=(1, 31))
        normed_eigs = eigs[1] / np.sqrt(eigs[0])

        # W_prime
        w_prime = (normed_eigs @ normed_eigs.T).astype(np.float32)

        return get_nKNN_pseudo_labels(w_prime, labeled_data, unlabeled_data, temperature=temperature)
