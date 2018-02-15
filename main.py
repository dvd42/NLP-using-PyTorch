import torch

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


import lstm as lstm
import data as data


# TODO: add args
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
    params = {'n_layers': 2, 'batch': 128, 'h_dim': 512,
              'seq': 65, 'type': dtype,
              'alphabet_size': len(dataset.alphabet)}

    dataloaders = data.loaders(dataset, params)

    rnn = lstm.LSTM(params).type(params['type'])

    optimizer = optim.Adam(rnn.parameters(), lr=0.095)
    criterion = nn.CrossEntropyLoss()

    char_to_ix = dataset.char_to_ix  # map from char to index
    rnn = lstm.train(dataloaders, char_to_ix, rnn, optimizer, criterion, params)

    # Get a batch from training set
    batch = dataloaders['train'].dataset[:params['batch'] * params['seq']]

    inputs = Variable(lstm.sequence_to_one_hot(batch, dataset.char_to_ix, params))

    ix_to_char = dataset.ix_to_char  # map from index to char

    string = rnn.gen_text(inputs[:, :-1, :], ix_to_char, iters=2, t=0.2)

    print(string)

    print(string, file=open('texts/output.txt', 'w'))


# Deberia quitar la primera capa linear?
# Deberia usar RELUs?
# Que proporcion deberia usar de validacion y training?
# He visto muchos ejemplos que usan RMSprop para estos casos, deberia usarlo?

"""
Esto es un fragmento de Graves et al.

A much larger network was used for this data than the Penn data (reflecting
the greater size and complexity of the training set) with seven hidden layers of
700 LSTM cells, giving approximately 21.3M weights. The network was trained
with stochastic gradient descent, using a learn rate of 0.0001 and a momentum
of 0.9. It took four training epochs to converge. The LSTM derivates were
clipped in the range [−1, 1].

A que se refiere con 7 capas de 700 LSTM cells? No seria eso 700 capas?

"""

if __name__ == "__main__":
    main()
