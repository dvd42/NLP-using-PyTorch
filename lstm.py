import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
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

        #self.n_params = alphabet_size*self.hidden_dim + self.hidden_dim*self.hidden_dim*self.n_layers + self.hidden_dim*output_size

        
        self.i2h = nn.Linear(alphabet_size,self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim,self.hidden_dim,self.n_layers,
                            batch_first=True,dropout=True)


        self.h2O = nn.Linear(self.hidden_dim, output_size)
        
        self.hidden = self.init_hidden(params['type'])
        
        
    def init_hidden(self,type):
       
        return (autograd.Variable(torch.zeros(self.n_layers, self.batch, 
            self.hidden_dim).type(type)),
                autograd.Variable(torch.zeros(self.n_layers, self.batch, 
                    self.hidden_dim).type(type)))


    def forward(self, sequence):
        out = self.i2h(sequence)
        lstm_out, self.hidden = self.lstm(out.view(self.batch,self.seq-1,-1),self.hidden)
        out = self.h2O(lstm_out.contiguous().view(-1,self.hidden_dim))
        return out
    
    
    def gen_text(self, out,ix_to_char,iters=2,t=None): 
        
        string = ''  
        self.train(False)

        for i in range(iters):
            
            out = self(out)
            _, idxs = out.max(1)
                        
            if t != None:
                
                soft_out = F.softmax(out/t,dim=1)
                p = soft_out.data.cpu().numpy()
            
                for j in range(soft_out.size()[0]):
                    
                    #print('Torch: {}'.format(torch.sum(soft_out.data[i])))
                    #print('Numpy: {}'.format(np.sum(soft_out.data.cpu().numpy()[i])))
                    
                    idxs[j] = np.random.choice(out.size()[1],p=p[j])
                    string += ix_to_char[idxs[j].data[0]] 
                                  
            else:
                for c in idxs.data:
                    string += ix_to_char[c]

        
        return string


def sequence_to_tensor(sequence,char_to_ix,params):
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
        print('='*15)

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
                
                inputs = autograd.Variable(
                    sequence_to_tensor(batch,char_map,params))
                  
                out = model(inputs[:,:-1,:])
                _,preds = out.max(1)

                _,target = inputs[:,1:,:].topk(1)
                

                loss = criterion(out,target.view(-1))

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == target).data[0]

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase,epoch_loss,epoch_acc))


            if phase == 'val': 

                # Save best weights
                if epoch_loss < best_loss:
                    bad_epochs = 0
                    best_loss = epoch_loss
                    #best_wts = model.state_dict()
                    torch.save(model.state_dict(),'rnn.pkl')
                    
                else:
                    bad_epochs += 1

             
        if bad_epochs == 10:
            break 
        
        epoch += 1

        #print(model.state_dict()['lstm.weight_ih_l0'])
        #model.load_state_dict(best_wts)


    time_elapsed = time.time() - since

    print('\nTraining completed in {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed % 60))

    print('Best Loss: {:.4f}'.format(best_loss))

        
    model.load_state_dict(torch.load('rnn.pkl'))
        
    return model