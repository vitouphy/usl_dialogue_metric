from collections import namedtuple
from datasets import VUPDataset
from models.VUPScorer import VUPScorer

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_vup(X_train, X_valid, args):
    '''
    X_train: an array of sentences.
    '''
    train_dataset = VUPDataset(X_train)
    valid_dataset = VUPDataset(X_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = VUPScorer(args).to(device)

    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    xtrain = [
        "i go to school.",
        "really? you don't like burger?"
    ]
    xvalid = [
        "i go to school.",
        "really? you don't like burger?"
    ]
    args = {
        "lr": 1e-5,
        "maxlen": 25,
        "batch_size": 16
    }
    args = namedtuple('args', args.keys())(*args.values())
    train_vup(xtrain, xvalid, args)
