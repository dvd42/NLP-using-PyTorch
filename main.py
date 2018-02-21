import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


import lstm as lstm
import data as data


# TODO: add args
# TODO: modify Txtloader to randomize sequences and get batches
# TODO: refactor main maybe
# TODO: add KeyboardInterrupt exception
def main():

    # Use GPU if available
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # path to the data file
    path = 'data/Lovecraft.txt'

    dataset = data.TxtLoader(path)

    # Network parameters
    params = {'n_layers': 2, 'batch': 100, 'h_dim': 512,
              'seq': 64, 'type': dtype,
              'alphabet_size': len(dataset.alphabet)}

    dataloaders = data.loaders(dataset, params)

    rnn = lstm.LSTM(params).type(params['type'])

    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()

    char_to_ix = dataset.char_to_ix  # map from char to index
    rnn = lstm.train(dataloaders, char_to_ix, rnn, optimizer, criterion, params)

    # Get a batch from training set
    batch = iter(dataloaders['train']).__next__()

    inputs = Variable(lstm.sequence_to_one_hot(batch, dataset.char_to_ix, params))

    ix_to_char = dataset.ix_to_char  # map from index to char

    string = rnn.gen_text(inputs[:, :-1, :], ix_to_char, iters=2, t=0.1)

    print(string)

    print(string, file=open('data/output.txt', 'w'))


if __name__ == "__main__":
    main()
