import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import time


class LSTM(nn.Module):

    def __init__(self,params):
        super(LSTM, self).__init__()
        
        self.hidden_dim = params['h_dim']
        self.n_layers = params['n_layers']
        self.batch = params['batch']
        self.seq = params['seq']
        alphabet_size = output_size = params['alphabet_size']

        
        self.i2h = nn.Linear(alphabet_size,self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim,self.hidden_dim,self.n_layers,
                            batch_first=True,dropout=True)

        self.h2O = nn.Linear(self.hidden_dim, output_size)
        
        self.hidden = self.init_hidden(params['type'])
        
        
    def init_hidden(self,type):
       
        return (Variable(torch.zeros(self.n_layers, self.batch, 
            self.hidden_dim).type(type)),
                Variable(torch.zeros(self.n_layers, self.batch, 
                    self.hidden_dim).type(type)))


    def forward(self, sequence):
        out = self.i2h(sequence)
        lstm_out, self.hidden = self.lstm(out.view(self.batch,self.seq-1,-1),self.hidden)
        out = self.h2O(out.contiguous().view(-1,self.hidden_dim))
        return out
    
    
    def gen_text(self, out,ix_to_char,iters=2,t=None): 
        
        string = ''  
        self.train(False)

        for i in range(iters):
            
            out = self(out)
            _, idxs = out.max(1)
                        
            if t != None:
                
                # Apply temperature
                soft_out = F.softmax(out/t,dim=1)
                p = soft_out.data.cpu().numpy()
                
                # Select a new predicted char with probability p
                for j in range(soft_out.size()[0]):
            
                    idxs[j] = np.random.choice(out.size()[1],p=p[j])
                    string += ix_to_char[idxs[j].data[0]] 
            
            # Select the predicted chars                     
            else:
                for c in idxs.data:
                    string += ix_to_char[c]

        
        return string


def sequence_to_one_hot(sequence,char_to_ix,params):

    tensor = torch.zeros(len(sequence),params['alphabet_size']).type(params['type'])

    for i, c in enumerate(sequence):
        tensor[i][char_to_ix[c]] = 1 

    return tensor.view(params['batch'],params['seq'],params['alphabet_size'])



def train(dataloaders,char_map,model,optimizer,criterion,params):
    
    since = time.time()

    best_loss = float('inf')
    epoch = 1
    bad_epochs = 0
    
    dataset_size = {x:len(dataloaders[x])*dataloaders[x].batch_size 
                        for x in ['train','val']}


    while True:
                
        print('Epoch {}'.format(epoch))
        print('='*10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            
            if phase == 'train':
                model.train(True) # Training Mode
            else:
                model.train(False) # Evaluate mode

            running_loss = 0
            running_corrects = 0

            # Iterate over the data
            for batch in dataloaders[phase]:

                model.zero_grad()
                model.hidden = model.init_hidden(params['type'])
                
                inputs = Variable(sequence_to_one_hot(batch,char_map,params))
                  
                out = model(inputs[:,:-1,:])
                _,preds = out.max(1)


                # Get the targets (indexes where the one-hot vector is 1)
                _,target = inputs[:,1:,:].topk(1)
                
                loss = criterion(out,target.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == target).data[0]

            # Compute mean epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase,epoch_loss,running_corrects))


            if phase == 'val': 

                # Save best weights
                if epoch_loss < best_loss:
                    bad_epochs = 0
                    best_loss = epoch_loss
                    #best_wts = model.state_dict()
                    torch.save(model.state_dict(),'rnn.pkl')
                    
                else:
                    bad_epochs += 1

        # Hara-kiri
        if bad_epochs == 10:
            break 
        
        epoch += 1


    time_elapsed = time.time() - since

    print('\nTraining completed in {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed % 60))

    print('Best Loss: {:.4f}\n\n'.format(best_loss))

    # Load best wts 
    model.load_state_dict(torch.load('rnn.pkl'))
        
    return model