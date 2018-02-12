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
    
    path = 'texts/Lovecraft.txt'
    dataset = data.TxtLoader(path)

    params = {'n_layers':1,'batch':128,
                'h_dim':512,'seq':100,'type':dtype,
                'alphabet_size':len(dataset.alphabet)}



    dataloaders = data.loaders(dataset,params)

    rnn = l.LSTM(params).type(params['type'])
    optimizer = optim.Adam(rnn.parameters(),lr=0.01)
    criterion = nn.CrossEntropyLoss()


    
    rnn = l.train(dataloaders,dataset.char_to_ix,rnn,optimizer,criterion,params)
    #print(rnn.state_dict()['lstm.weight_ih_l0'])


    batch = dataloaders['train'].dataset[:params['batch']*params['seq']]


    inputs = autograd.Variable(
        l.sequence_to_tensor(batch,dataset.char_to_ix,params))

    string = rnn.gen_text(inputs[:,:-1,:],dataset.ix_to_char,iters=2)

    print(string)           


    print(string,file=open('texts/output.txt','w'))


# FIXME: why is temperature so bad?
# TODO: add RELUs?
# FIXME: hyperparameters

if __name__ == "__main__":
    main()


