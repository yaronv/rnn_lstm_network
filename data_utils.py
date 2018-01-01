import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def prepare_tensor_dataset(fname, workers, batch_size):
    X, Y = prepare_data(fname)

    X = (torch.from_numpy(X)).type(torch.FloatTensor)
    Y = (torch.from_numpy(Y)).type(torch.LongTensor)

    set = TensorDataset(X, Y)
    set_data_loader = DataLoader(dataset=set, num_workers=workers, batch_size=batch_size, shuffle=True)
    return set_data_loader


def prepare_data(fname):
    with open(fname) as f:
        content = f.readlines()

    X = []
    Y = []
    for line in content:
        if line != '\n':
            line = line.strip()
            x = list(line.split()[0])
            x = [unicode_to_ascii(s) for s in x]
            y = int(line.split()[1])
            X.append(np.asarray(x))
            Y.append(y)

            zipped = zip(np.asarray(X), np.asarray(Y))
            zipped.sort(key=lambda x: len(x[0]), reverse=True)
            X_sorted, Y_sorted = zip(*(zipped))

            max_len = len(X_sorted[0])

            X_padded = [np.append(x, [0]*(max_len - len(x))) for x in X_sorted]

    return np.asarray(X_padded), np.asarray(Y_sorted)


def get_output_vector_from_value(value):
    if value == 1:
        return np.asarray([1, 0])
    else:
        return np.asarray([0, 1])


def unicode_to_ascii(s):
    return ord(s)


def ascii_to_unicode(s):
    return chr(s)
