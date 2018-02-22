import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class LSTM(nn.Module):
    """LSTM neural network

    Args:
         params (dict): holds the program hyperparameters
    """

    def __init__(self, params):
        super(LSTM, self).__init__()

        self.hidden_dim = params['nhid']
        self.n_layers = params['nlayers']
        self.batch = params['batch']
        self.seq = params['seq']
        self.dropout = params['dropout']
        alphabet_size = self.output_size = params['alphabet_size']

        self.lstm = nn.LSTM(alphabet_size, self.hidden_dim, self.n_layers,
                            batch_first=True, dropout=self.dropout)

        self.h2O = nn.Linear(self.hidden_dim, self.output_size)
        self.hidden = self.init_hidden(params['type'])

    def init_hidden(self, type):
        """Initialize the LSTM hidden and cell state

        Args:
            type: the tensor type e.g:torch.FloatTensor, torch.cuda.FloatTensor

        Returns:
            h_0,c_0 (Variable,Variable): Tensors of size (L,B,H) where:
            L: number of LSTM layers
            B: batch size
            H: hidden dimension of the lstm
        """
        h_0 = Variable(
            torch.zeros(self.n_layers, self.batch, self.hidden_dim).type(type))

        c_0 = Variable(
            torch.zeros(self.n_layers, self.batch, self.hidden_dim).type(type))

        return h_0, c_0

    def count_parameters(self):
        """Counts the neural net parameters

        Returns:
            (int): the amount of parameters in the neural net
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, sequence):
        """Computes the neural net forward pass

        Args:
            sequence (Variable): one-hot Tensor of size (B,SL,AS) where:
            B: batch size
            SL: sequence lenght
            AS: alphabet size

        Returns:
            out (Variable): one-hot Tensor of size (B*SL,AS)

        """

        out = sequence.contiguous().view(self.batch, self.seq, -1)
        lstm_out, self.hidden = self.lstm(out, self.hidden)
        out = self.h2O(lstm_out)
        return out.view(-1, self.output_size)

    def gen_text(self, out, ix2char, nchars, t=0.5):
        """Reproduces text using the LSTM

        Args:
            out (Variable): one-hot Tensor of size (B,SL,AS) where:
            B: batch size
            SL: sequence lenght
            AS: alphabet size

            ix2char (dict): mapping from integers (indexes) to chars

            nchars (int): number of chars to be generated.
            t (float,optional): softmax temperature value. Default: 0.5

        Returns:
            (str): generated text
        """

        string = ''
        self.eval()

        while len(string) < nchars:

            out = self(out)
            _, idxs = out.max(1)

            # Apply temperature
            soft_out = F.softmax(out / t, dim=1)
            p = soft_out.data.cpu().numpy()

            # Select a new predicted char with probability p
            for j in range(soft_out.size()[0]):

                idxs[j] = np.random.choice(out.size()[1], p=p[j])
                string += ix2char[idxs[j].data[0]]

        return string
