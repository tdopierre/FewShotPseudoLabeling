import json


def load_data_jsonl(file_path):
    out = list()
    with open(file_path, 'r') as file:
        for line in file:
            out.append(json.loads(line.strip()))
    return out


def save_data_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for d in data:
            file.write(json.dumps(d, ensure_ascii=False) + '\n')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Vocab:
    def __init__(self, labels_list):
        uniq_labels_list = sorted(set(labels_list))
        self._str_to_ix = {l: ix for ix, l in enumerate(uniq_labels_list)}
        self._ix_to_str = {ix: l for ix, l in enumerate(uniq_labels_list)}
        self.labels = uniq_labels_list

    def __call__(self, x, rev=False):
        if rev:
            return self._ix_to_str[x]
        else:
            return self._str_to_ix[x]
