
# coding: utf-8

# In[1]:


import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import random


# In[2]:


if torch.cuda.is_available:
    DTYPE = torch.cuda.FloatTensor
else:
    DTYPE = torch.FloatTensor
    


# In[3]:


data = open('texts/Lovecraft.txt','r').read().lower()
alphabet = set(data)

ix_to_char = {k:v for k,v in enumerate(alphabet)}
char_to_ix = {k:v for v,k in enumerate(alphabet)}


# In[4]:


# Define function to prepare sequences

def prepare_seq(data, drop_last=False):
    
    sequences = []
    
    # Create (len(data)/SEQ_LEN+1) number of vectors of SEQ_LEN lenght
    for i in range(0,len(data),SEQ_LEN+1):

        chars = [char_to_ix[c] for c in data[i:i+SEQ_LEN+1]]
        sequences.append(chars)
    
    # Drop last batch if incomplete`
    if drop_last and len(sequences) % BATCH_SIZE != 0:
        
        index = len(sequences)//BATCH_SIZE * BATCH_SIZE
        del(sequences[index:])
    
    
    # Drop last sequence if incomplete
    elif len(sequences[-1]) != SEQ_LEN:
        del(sequences[-1])
    
    sequences = np.array([sequences]).reshape((-1,SEQ_LEN+1))
    
    # Create inputs and targets
    inputs = sequences[:,:-1]
    targets = sequences[:,1:]
    
    
    # Convert sequences to variables
    inputs =  autograd.Variable(torch.Tensor(inputs).type(DTYPE))
    targets = autograd.Variable(torch.Tensor(targets).type(DTYPE))
    
    
    return inputs, targets


# In[5]:


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_dim,hidden_dim2 ,output_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        #self.hidden_dim2 = hidden_dim2
        self.lstm = nn.LSTM(input_size,hidden_dim,NUM_LAYERS)
        #self.lstm2 = nn.LSTM(hidden_dim,hidden_dim2)
        self.h2O = nn.Linear(hidden_dim, output_size)
        self.hidden = self.init_hidden(self.hidden_dim)
        #self.hidden2 = self.init_hidden(self.hidden_dim2)
        
        
    def init_hidden(self,hidden_dim):
       
        return (autograd.Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, hidden_dim).type(DTYPE)),
                autograd.Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, hidden_dim).type(DTYPE)))

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence,self.hidden)
        #lstm_out, self.hidden2 = self.lstm2(lstm_out,self.hidden2)
        out = self.h2O(lstm_out.view(-1,self.hidden_dim))
        return out


# In[6]:


NUM_LAYERS = 1
BATCH_SIZE = 128
HIDDEN_DIM = 128
HIDDEN_DIM2 = 128
SEQ_LEN = 64
inputs,targets = prepare_seq(data[:1000000],drop_last=True)

input_size = 1


# In[7]:


rnn = LSTM(input_size,HIDDEN_DIM,HIDDEN_DIM2,len(alphabet)).type(DTYPE)
optimizer = optim.Adam(rnn.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 1000


# In[8]:


for epoch in range(epochs):
    
    losses = np.array([])
    
    permutation = torch.randperm(inputs.size()[0]).type(DTYPE).long()    
    
    for i in range(0,inputs.size()[0],BATCH_SIZE):
        
        rnn.zero_grad()
        rnn.hidden = rnn.init_hidden(rnn.hidden_dim)
        #rnn.hidden2 = rnn.init_hidden(rnn.hidden_dim2)
        
        
        idxs = permutation[i:i+BATCH_SIZE]   
        out = rnn(inputs[idxs].view(SEQ_LEN,BATCH_SIZE,-1))    
        
        loss = criterion(out,targets[idxs].view(-1).long())
        losses = np.append(losses,loss.data[0])
        
        
        loss.backward()
        optimizer.step()


    print("Epoch {}/{}\nLoss: {:.2f}".format(epoch+1,epochs,losses.mean()))
    print("="*15)

    


# In[9]:


string = ''  
temperature = 0.5

#idxs = permutation[i:i+BATCH_SIZE]

for i in range(0,inputs.size()[0],BATCH_SIZE):
    
    out = rnn(inputs[i:i+BATCH_SIZE,:].view(SEQ_LEN,BATCH_SIZE,-1))
    _ ,ix = out.topk(5)
    
    for j in range(ix.shape[0]):
    
        t = random.random()

        if t < temperature:
            selection = random.randint(1,ix.shape[1]-1)
        else:
            selection = 0
        
        string += ix_to_char[ix[j,selection].data[0]]

print(string)       
#print(string,file=open('texts/output.txt','w'))
    
    #with open('texts/output.txt', mode='wt', encoding='utf-8') as myfile:
        #myfile.writelines(' '.join(lyrics))

