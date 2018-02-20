import torch.utils.data as data


class TxtLoader(data.Dataset):
    """Takes a .txt file and turns its contents into a dataset

    Args:
        path (str): The path to the .txt file holding the data
    """

    def __init__(self, path):
        super(TxtLoader, self).__init__()
        self.text = open(path, 'r').read()
        self.alphabet = set(self.text)
        self.ix_to_char = {k: v for k, v in enumerate(self.alphabet)}
        self.char_to_ix = {k: v for v, k in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index]


def loaders(dataset, params):
    """Creates a PyTorch Dataloader for training and one for validation

        Args:
            dataset (TxtLoader): dataset holding the data
            params (dict): holds the program hyperparameters

        Returns:
            dataloaders (dict): holds PyTorch Dataloaders for training and validation
    """

    # 80% for training and the rest for validation
    train_idx = int(len(dataset) * 0.8)
    datasets = {'train': dataset[:train_idx],
                'val': dataset[train_idx:]}

    # Create the DataLoaders
    dataloaders = {x: data.DataLoader(datasets[x],
                                      batch_size=params['batch'] * (params['seq'] + 1),
                                      drop_last=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders
