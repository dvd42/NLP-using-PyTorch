import torch.utils.data as data


class TxtLoader(data.Dataset):
    
    def __init__(self,path):
        super(TxtLoader,self).__init__()
        self.text = open(path,'r').read()
        self.alphabet = set(self.text)
        self.ix_to_char = {k:v for k,v in enumerate(self.alphabet)}
        self.char_to_ix = {k:v for v,k in enumerate(self.alphabet)}


    def __len__(self):
        return len(self.text)
    
    def __getitem__(self,index):
        return self.text[index]



def loaders(dataset,params): 

	# 80% for training and the rest for validation
	train_idx = int(len(dataset)*.8)+1
	datasets = {'train':dataset[:train_idx],
				'val':dataset[train_idx:]
				}
				
	# Create the DataLoaders
	dataloaders = {x: data.DataLoader(datasets[x],batch_size=params['batch']*params['seq'],
	                                  drop_last=True,num_workers=4)
	                for x in ['train','val']}



	return dataloaders