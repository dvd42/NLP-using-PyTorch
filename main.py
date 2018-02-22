import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse

import lstm
import data

parser = argparse.ArgumentParser(description='PyTorch char-level LSTM')
parser.add_argument('--data', type=str, default='data/Lovecraft.txt',
                    help='location of the data corpus')
parser.add_argument('--nhid', type=int, default=512,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of LSTM layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clget main threadipping')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--seq', type=int, default=64,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--temp', type=float, default=0.5,
                    help='softmax temperature')
parser.add_argument('--nchars', type=int, default=3000,
                    help='number of chars to generate')

args = parser.parse_args()

dtype = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

path = args.data
dataset = data.TxtLoader(path)

params = {'nhid': args.nhid, 'nlayers': args.nlayers, 'dropout': args.dropout,
          'batch': args.batch_size, 'seq': args.seq, 'type': dtype,
          'alphabet_size': len(dataset.alphabet)}

dataloaders = data.loaders(dataset, params)
model = lstm.LSTM(params).type(params['type'])
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


def sequence_to_one_hot(sequence):
    """Turns a sequence of chars into one-hot Tensor"""

    batch_size = params['batch'] * (params['seq'] + 1)
    assert len(sequence) == batch_size, 'Sequence must be a batch'

    tensor = torch.zeros(len(sequence), params['alphabet_size']).type(params['type'])

    for i, c in enumerate(sequence):
        tensor[i][dataset.char2ix[c]] = 1

    return tensor.view(params['batch'], params['seq'] + 1, params['alphabet_size'])


def train():
    """Trains the neural net"""

    model.train()
    running_loss = 0

    # Iterate over the data
    for batch in dataloaders['train']:

        model.hidden = model.init_hidden(params['type'])
        model.zero_grad()  # Center gradient

        inputs = Variable(sequence_to_one_hot(batch))
        out = model(inputs[:, :-1, :])
        _, preds = out.max(1)

        # Get the targets (indexes where the one-hot vector is 1)
        _, targets = inputs[:, 1:, :].topk(1)
        loss = criterion(out, targets.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        running_loss += loss.data[0]

    # Compute mean epoch loss
    epoch_loss = running_loss / len(dataloaders['train'])

    return epoch_loss


def evaluate(bad_epochs):
    """Evaluates the model"""

    model.eval()  # Evaluate Mode
    running_loss = 0

    # Iterate over the data
    for batch in dataloaders['val']:

        model.hidden = model.init_hidden(params['type'])
        model.zero_grad()

        inputs = Variable(sequence_to_one_hot(batch), volatile=True)

        out = model(inputs[:, :-1, :])
        _, preds = out.max(1)

        # Get the targets (indexes where the one-hot vector is 1)
        _, targets = inputs[:, 1:, :].topk(1)
        loss = criterion(out, targets.view(-1))

        running_loss += loss.data[0]

    # Compute mean epoch loss
    epoch_loss = running_loss / len(dataloaders['val'])

    return epoch_loss, bad_epochs


epoch = 1
bad_epochs = 0
best_val_loss = float('inf')

try:
    while True:
        epoch_start_time = time.time()

        train_loss = train()
        print('-' * 53)
        print('| End of epoch: {:2d} | Time: {:5.2f}s | Train loss: {:.2f}|'
              .format(epoch, (time.time() - epoch_start_time), train_loss))

        val_loss, bad_epochs = evaluate(bad_epochs)
        print('-' * 53)
        print('| End of epoch: {:2d} | Time: {:5.2f}s | Valid loss: {:.2f}|'
              .format(epoch, (time.time() - epoch_start_time), val_loss))

        # Save the best model so far
        if val_loss < best_val_loss:

            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            bad_epochs = 0

        else:
            bad_epochs += 1

        # Hara-kiri
        if bad_epochs == 10:
            break

        epoch += 1

except KeyboardInterrupt:
    print('-' * 53)
    print('Exiting from training early')

print('Best Loss: {}'.format(best_val_loss))
print('=' * 53 + '\n')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Get a batch from training set
batch = iter(dataloaders['train']).__next__()

inputs = Variable(sequence_to_one_hot(batch), volatile=True)

string = model.gen_text(inputs[:, :-1, :], dataset.ix2char, args.nchars, t=args.temp)

print(string)
