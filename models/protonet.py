import os
import fasttext
import tqdm
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange
from time import sleep
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from transformers import BertModel, BertTokenizer, BertConfig, AutoConfig, AutoModel, AutoTokenizer
import warnings
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    def loss(self, sample):
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        x = [item for xs_ in xs for item in xs_] + [item for xq_ in xq for item in xq_]
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        dists = euclidean_dist(zq, z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        dists.view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'dists': dists,
            'target': target_inds
        }

    def loss_softkmeans(self, sample):
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        xu = sample['xu']  # unlabeled

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        x = [item for xs_ in xs for item in xs_] + [item for xq_ in xq for item in xq_] + [item for item in xu]
        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        zs = z[:n_class * n_support]
        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support: (n_class * n_support) + (n_class * n_query)]
        zu = z[(n_class * n_support) + (n_class * n_query):]

        distances_to_proto = euclidean_dist(
            torch.cat((zs, zu)),
            z_proto
        )

        distances_to_proto_normed = torch.nn.Softmax(dim=-1)(-distances_to_proto)

        refined_protos = list()
        for class_ix in range(n_class):
            z = torch.cat(
                (zs[class_ix * n_support: (class_ix + 1) * n_support], zu)
            )
            d = torch.cat(
                (torch.ones(n_support).to(device),
                 distances_to_proto_normed[(n_class * n_support):, class_ix])
            )
            refined_proto = ((z.t() * d).sum(1) / d.sum())
            refined_protos.append(refined_proto.view(1, -1))
        refined_protos = torch.cat(refined_protos)

        dists = euclidean_dist(zq, refined_protos)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        dists.view(n_class, n_query, -1)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'dists': dists,
            'target': target_inds
        }


def train_step(protonet: Protonet, optimizer, train_data_lab_to_sentences_dict, Ns=2, Nc=3, Nq=2, refined=False):
    protonet.train()
    import random
    Nc = min(Nc, len(train_data_lab_to_sentences_dict.keys()))
    rand_keys = np.random.choice(list(train_data_lab_to_sentences_dict.keys()), Nc, replace=False)

    for key, val in train_data_lab_to_sentences_dict.items():
        random.shuffle(val)

    sample = {
        "xs": [
            [train_data_lab_to_sentences_dict[k][i] for i in range(Ns)] for k in rand_keys
        ],
        "xq": [
            [train_data_lab_to_sentences_dict[k][Ns + i] for i in range(Nq)] for k in rand_keys
        ]
    }

    if refined:
        sample['xu'] = [
            item for k in rand_keys for item in train_data_lab_to_sentences_dict[k][Ns + Nq:Ns + Nq + 10]
        ]

    optimizer.zero_grad()
    torch.cuda.empty_cache()
    if refined:
        loss, loss_dict = protonet.loss_softkmeans(sample)
    else:
        loss, loss_dict = protonet.loss(sample)
    loss.backward()
    optimizer.step()

    return loss, loss_dict


def test_step(protonet: Protonet, data_lab_to_sentences_dict, Ns=2, Nc=3, Nq=2, refined=False):
    accs = list()
    protonet.eval()
    for i in range(100):
        import random
        Nc = min(Nc, len(data_lab_to_sentences_dict.keys()))
        rand_keys = np.random.choice(list(data_lab_to_sentences_dict.keys()), Nc, replace=False)

        for key, val in data_lab_to_sentences_dict.items():
            random.shuffle(val)

        sample = {
            "xs": [
                [data_lab_to_sentences_dict[k][i] for i in range(Ns)] for k in rand_keys
            ],
            "xq": [
                [data_lab_to_sentences_dict[k][Ns + i] for i in range(Nq)] for k in rand_keys
            ]
        }

        if refined:
            sample['xu'] = [
                item for k in rand_keys for item in data_lab_to_sentences_dict[k][Ns + Nq:Ns + Nq + 10]
            ]

        with torch.no_grad():
            if refined:
                loss, loss_dict = protonet.loss_softkmeans(sample)
            else:
                loss, loss_dict = protonet.loss(sample)

        accs.append(loss_dict["acc"])

    return np.mean(accs)


def load_weights(filename, protonet, use_gpu):
    if use_gpu:
        protonet.load_state_dict(torch.load(filename))
    else:
        protonet.load_state_dict(torch.load(filename), map_location="cpu")
    return protonet


def run_proto(train_path, model_name_or_path, test_input_path=None, test_output_path=None, refined=False):
    import numpy as np
    from util.data import get_jsonl_data
    import random
    import collections
    import os
    import pickle

    if test_output_path:
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    if test_input_path:
        test_sentences = [line.strip() for line in open(test_input_path, 'r').readlines() if len(line.strip())]
    # train_path = f'data/datasets/Liu/few-shot_final/01/train.jsonl'

    # Load model
    bert = BERTEncoder(model_name_or_path).to(device)
    net = Protonet(encoder=bert)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-5)

    # Load data
    data = get_jsonl_data(train_path)
    print("Data loaded")
    data_dict = collections.defaultdict(list)
    for d in data:
        data_dict[d['label']].append(d['sentence'])
    data_dict = dict(data_dict)
    for k, d in data_dict.items():
        random.shuffle(d)

    labels = sorted(data_dict.keys())
    random.shuffle(labels)
    labels_train = labels[:int(len(labels) / 2)]
    labels_valid = labels[int(len(labels) / 2):]

    print(f"Train Labels ({len(labels_train)}) {labels_train}")
    print(f"Valid Labels ({len(labels_valid)}) {labels_valid}")

    # train_data_dict = {
    #     k: d[:int(0.7 * len(d))] for k, d in data_dict.items()
    # }
    # valid_data_dict = {
    #     k: d[int(0.7 * len(d)):] for k, d in data_dict.items()
    # }
    train_data_dict = {label: data_dict[label] for label in labels_train}
    valid_data_dict = {label: data_dict[label] for label in labels_valid}

    print("Data split. starting training")

    accs = list()
    n_eval_since_last_best = 0
    best_valid_acc = 0.0

    for _ in range(10000):
        loss, loss_dict = train_step(net, optimizer, train_data_dict, refined=refined)
        accs.append(loss_dict['acc'])
        if (_ + 1) % 100 == 0:
            train_acc = np.mean(accs)
            valid_acc = test_step(net, valid_data_dict, refined=refined)
            if valid_acc > best_valid_acc:
                print(f"Train acc={train_acc:.4f} | Valid acc={valid_acc:.4f} (better)")
                n_eval_since_last_best = 0
                best_valid_acc = valid_acc

                if test_input_path:
                    embeddings = list()
                    for i in tqdm.tqdm(range(0, len(test_sentences), 16)):
                        net.eval()
                        with torch.no_grad():
                            embeddings.append(net.encoder.forward(test_sentences[i:i + 16]).cpu().detach().numpy())
                    with open(test_output_path, "wb") as file:
                        pickle.dump(embeddings, file)
            else:
                n_eval_since_last_best += 1
                print(f"Train acc={train_acc:.4f} | Valid acc={valid_acc:.4f} (worse, {n_eval_since_last_best})")
        if n_eval_since_last_best >= 5:
            print(f"Early-stopping.")
            break


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", type=str, required=True)
    parser.add_argument("--test-input-file-path", type=str)
    parser.add_argument("--test-output-file-path", type=str)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--refined", action="store_true")

    args = parser.parse_args()
    logger.debug(f"Received args {args}")

    if args.test_output_file_path and os.path.exists(args.test_output_file_path):
        print(f"[OK] {args.test_output_file_path}")
        exit(0)

    if args.test_input_file_path and not os.path.exists(args.test_input_file_path):
        raise FileNotFoundError(f'{args.test_input_file_path} not found.')
    if args.train_file_path and not os.path.exists(args.train_file_path):
        raise FileNotFoundError(f'{args.train_file_path} not found')

    run_proto(
        train_path=args.train_file_path,
        test_input_path=args.test_input_file_path,
        test_output_path=args.test_output_file_path,
        model_name_or_path=args.model_name_or_path,
        refined=args.refined
    )


if __name__ == '__main__':
    main()
