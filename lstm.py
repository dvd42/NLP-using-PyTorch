import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import time


class LSTM(nn.Module):

    """LSTM neural network

    Args:
         params (Dicts): holds the program hyperparameters
    """

    def __init__(self, params):
        super(LSTM, self).__init__()

        self.hidden_dim = params['h_dim']
        self.n_layers = params['n_layers']
        self.batch = params['batch']
        self.seq = params['seq']
        alphabet_size = output_size = params['alphabet_size']

        self.i2h = nn.Linear(alphabet_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers,
                            batch_first=True, dropout=True)

        self.h2O = nn.Linear(self.hidden_dim, output_size)
        self.hidden = self.init_hidden(params['type'])

    def init_hidden(self, type):
        """Initialize the LSTM hidden and cell state

        Args:
            type: the tensor type e.g:torch.FloatTensor, torch.cuda.FloatTensor

        Returns:
            (Variable,Variable): Tensors of size (L,B,H) where:
            L: number of LSTM layers
            B: batch size
            H: hidden dimension of the lstm
        """
        h_0 = Variable(
            torch.zeros(self.n_layers, self.batch, self.hidden_dim).type(type))

        c_0 = Variable(
            torch.zeros(self.n_layers, self.batch, self.hidden_dim).type(type))

        return h_0, c_0

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

        out = self.i2h(sequence)
        lstm_out, self.hidden = self.lstm(
            out.view(self.batch, self.seq - 1, -1), self.hidden)
        out = self.h2O(out.contiguous().view(-1, self.hidden_dim))
        return out

    def gen_text(self, out, ix_to_char, iters=2, t=None):
        """Reproduces text using the LSTM

        Args:
            out (Variable): one-hot Tensor of size (B,SL,AS) where:
            B: batch size
            SL: sequence lenght
            AS: alphabet size

            ix_to_char (Dict): mapping from integers (indexes) to chars

            iters (int): number of text sequences to be generated. Default: 2
            t (float): softmax temperature value (applied if not None). Default: None

        Returns:
            (String): generated text
        """

        string = ''
        self.train(False)

        for i in range(iters):

            out = self(out)
            _, idxs = out.max(1)

            if t is not None:

                # Apply temperature
                soft_out = F.softmax(out / t, dim=1)
                p = soft_out.data.cpu().numpy()

                # Select a new predicted char with probability p
                for j in range(soft_out.size()[0]):

                    idxs[j] = np.random.choice(out.size()[1], p=p[j])
                    string += ix_to_char[idxs[j].data[0]]

            # Select the predicted chars
            else:
                for c in idxs.data:
                    string += ix_to_char[c]

        return string


def sequence_to_one_hot(sequence, char_to_ix, params):
    """Turns a sequence of chars into one-hot Tensor

    Args:
        sequence (String): sequence of chars
        char_to_ix (Dict): mapping from chars to integers (indexes)
        params (Dicts): holds the program hyperparameters

    Returns:
        (Tensor): one-hot tensor of size (B,SL,AS) where:
        B: batch size
        SL: sequence lenght
        AS: alphabet size
    """

    batch_size = params['batch'] * params['seq']
    assert len(sequence) == batch_size, 'Sequence must be a batch'

    tensor = torch.zeros(len(sequence), params['alphabet_size']).type(params['type'])

    for i, c in enumerate(sequence):
        tensor[i][char_to_ix[c]] = 1

    return tensor.view(params['batch'], params['seq'], params['alphabet_size'])


def train(dataloaders, char_to_ix, model, optimizer, criterion, params):
    """Trains the neural net

    Args:
        dataloaders (Dict): holds PyTorch Dataloaders for training and validation
        char_to_ix: (Dict): mapping from chars to integers (indexes)
        model (LSTM): the model to be trained
        optimizer (Optimizer): PyTorch optimizer to use
        criterion: Loss function to use
        params (Dicts): holds the program hyperparameters

        Returns:
            model (LSTM): the trained model
    """

    since = time.time()

    best_loss = float('inf')
    epoch = 1
    bad_epochs = 0

    # dataset_size = {x: len(dataloaders[x]) * dataloaders[x].batch_size
    # for x in ['train', 'val']}

    while True:
        print('Epoch {}'.format(epoch))
        print('=' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train(True)  # Training Mode
            else:
                model.train(False)  # Evaluate mode

            running_loss = 0
            running_corrects = 0

            # Iterate over the data
            for batch in dataloaders[phase]:

                model.zero_grad()
                model.hidden = model.init_hidden(params['type'])

                inputs = Variable(sequence_to_one_hot(batch, char_to_ix, params))

                out = model(inputs[:, :-1, :])
                _, preds = out.max(1)

                # Get the targets (indexes where the one-hot vector is 1)
                _, target = inputs[:, 1:, :].topk(1)

                loss = criterion(out, target.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == target).data[0]

            # Compute mean epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase])
            # epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, running_corrects))

            if phase == 'val':

                # Save best weights
                if epoch_loss < best_loss:
                    bad_epochs = 0
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'rnn.pkl')

                else:
                    bad_epochs += 1

        # Hara-kiri
        if bad_epochs == 10:
            break

        epoch += 1

    time_elapsed = time.time() - since

    print('\nTraining completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best Loss: {:.4f}\n\n'.format(best_loss))

    # Load best wts
    model.load_state_dict(torch.load('rnn.pkl'))

    return model
