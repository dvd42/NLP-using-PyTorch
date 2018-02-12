import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np


import lstm as l
import data as data



def main():

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    path = 'texts/shk.txt'
    dataset = data.TxtLoader(path)

    params = {'n_layers':1,'batch':128,
                'h_dim':512,'seq':51,'type':dtype,
                'alphabet_size':len(dataset.alphabet)}

    dataloaders = data.loaders(dataset,params)

    rnn = l.LSTM(params).type(params['type'])

    optimizer = optim.Adam(rnn.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()


    char_to_ix = dataset.char_to_ix
    rnn = l.train(dataloaders,char_to_ix,rnn,optimizer,criterion,params)
    #print(rnn.state_dict()['lstm.weight_ih_l0'])


    batch = dataloaders['train'].dataset[:params['batch']*params['seq']]


    inputs = autograd.Variable(
        l.sequence_to_tensor(batch,dataset.char_to_ix,params))


    ix_to_char = dataset.ix_to_char
    string = rnn.gen_text(inputs[:,:-1,:],ix_to_char,iters=2)

    print(string)           


    print(string,file=open('texts/output.txt','w'))

# TODO: add args
# TODO: add RELUs?
# TODO: repeat tests
# FIXME: hyperparameters


# train and val proportion?
# rmsProp?
# primetext?
# bilinear?
# hidden dim?
# LSTM cell sequence?

"""
A much larger network was used for this data than the Penn data (reflecting
the greater size and complexity of the training set) with seven hidden layers of
700 LSTM cells, giving approximately 21.3M weights. The network was trained
with stochastic gradient descent, using a learn rate of 0.0001 and a momentum
of 0.9. It took four training epochs to converge. The LSTM derivates were
clipped in the range [−1, 1]. ???
"""

if __name__ == "__main__":
    main()


