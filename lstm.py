import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import time


class LSTM(nn.Module):
    """LSTM neural network

    Args:
         params (dict): holds the program hyperparameters
    """

    def __init__(self, params):
        super(LSTM, self).__init__()

        self.hidden_dim = params['h_dim']
        self.n_layers = params['n_layers']
        self.batch = params['batch']
        self.seq = params['seq']
        alphabet_size = self.output_size = params['alphabet_size']

        self.i2h = nn.Linear(alphabet_size, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers,
                            batch_first=True, dropout=True)

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
        out = out.view(self.batch, self.seq, -1)
        lstm_out, self.hidden = self.lstm(out, self.hidden)
        out = self.h2O(lstm_out)
        return out.view(-1, self.output_size)

    def gen_text(self, out, ix_to_char, iters=2, t=1):
        """Reproduces text using the LSTM

        Args:
            out (Variable): one-hot Tensor of size (B,SL,AS) where:
            B: batch size
            SL: sequence lenght
            AS: alphabet size

            ix_to_char (dict): mapping from integers (indexes) to chars

            iters (int,optional): number of tequences to be generated. Default: 2
            t (float,optional): softmax temperature value. Default: 1

        Returns:
            (str): generated text
        """

        string = ''
        self.train(False)

        for i in range(iters):

            out = self(out)
            _, idxs = out.max(1)

            # Apply temperature
            soft_out = F.softmax(out / t, dim=1)
            p = soft_out.data.cpu().numpy()

            # Select a new predicted char with probability p
            for j in range(soft_out.size()[0]):

                idxs[j] = np.random.choice(out.size()[1], p=p[j])
                string += ix_to_char[idxs[j].data[0]]

        return string


def sequence_to_one_hot(sequence, char_to_ix, params):
    """Turns a sequence of chars into one-hot Tensor

    Args:
        sequence (str): sequence of chars
        char_to_ix (dict): mapping from chars to integers (indexes)
        params (dict): holds the program hyperparameters

    Returns:
        (Tensor): one-hot tensor of size (B,SL+1,AS) where:
        B: batch size
        SL: sequence lenght
        AS: alphabet size
    """

    batch_size = params['batch'] * (params['seq'] + 1)
    assert len(sequence) == batch_size, 'Sequence must be a batch'

    tensor = torch.zeros(len(sequence), params['alphabet_size']).type(params['type'])

    for i, c in enumerate(sequence):
        tensor[i][char_to_ix[c]] = 1

    return tensor.view(params['batch'], params['seq'] + 1, params['alphabet_size'])


def train(dataloaders, char_to_ix, model, optimizer, criterion, params):
    """Trains the neural net

    Args:
        dataloaders (dict): holds PyTorch Dataloaders for training and validation
        char_to_ix: (dict): mapping from chars to integers (indexes)
        model (LSTM): the model to be trained
        optimizer (Optimizer): PyTorch optimizer
        criterion: Loss function
        params (dict): holds the program hyperparameters

        Returns:
            model (LSTM): the trained model
    """

    assert len(dataloaders['train']) != 0, 'Not enough data for training'
    assert len(dataloaders['val']) != 0, 'Not enough data for validation'

    since = time.time()

    best_acc = 0
    epoch = 1
    bad_epochs = 0

    dataset_size = {x: len(dataloaders[x]) * dataloaders[x].batch_size
                    for x in ['train', 'val']}

    print(dataset_size)

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
                targets = inputs[:, 1:, :].topk(1)[1].view(-1)
                loss = criterion(out, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == targets).data[0]

            # Compute mean epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':

                # Save best weights
                if epoch_acc > best_acc:
                    bad_epochs = 0
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), 'rnn.pkl')

                else:
                    bad_epochs += 1

        # Hara-kiri
        if bad_epochs == 20:
            break

        epoch += 1

    time_elapsed = time.time() - since

    print('\nTraining completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best Accuracy: {:.4f}\n\n'.format(best_acc))

    # Load best wts
    model.load_state_dict(torch.load('rnn.pkl'))

    return model
