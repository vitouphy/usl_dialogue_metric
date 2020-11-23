from collections import namedtuple
from datasets import VUPDataset, NUPDataset, MLMDataset
from models.VUPScorer import VUPScorer
from models.NUPScorer import NUPScorer
from models.MLMScorer import MLMScorer

import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, X_train, X_valid):

    if args.metric == "VUP":
        train_dataset = VUPDataset(X_train, maxlen=args.res_token_len)
        valid_dataset = VUPDataset(X_valid, maxlen=args.res_token_len)
        model = VUPScorer(args).to(device)

    elif args.metric == "NUP":
        train_dataset = NUPDataset(X_train, ctx_token_len=args.ctx_token_len, res_token_len=args.res_token_len)
        valid_dataset = NUPDataset(X_valid, ctx_token_len=args.ctx_token_len, res_token_len=args.res_token_len)
        model = NUPScorer(args).to(device)

    elif args.metric == "MLM":
       train_dataset = MLMDataset(X_train, maxlen=args.res_token_len)
       valid_dataset = MLMDataset(X_valid, maxlen=args.res_token_len)
       model = MLMScorer(args).to(device)

    else:
        raise Exception('Please select model from the following. VUP|NUP|MLM')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = pl.Trainer(max_epochs=args.max_epochs, weights_save_path=args.weight_path)
    trainer.fit(model, train_dataloader, valid_dataloader)

    print ("[!] training complete")


def read_dataset(path):
    '''
    - A line of sentence x -> Output a of sentence x
    - A line of 2 sentences separated by tab -> a tuple of (c,r)
    '''
    arr = []
    with open(path) as f:
        for line in f:
            sents = [ x.strip() for x in line.split('\t') ]
            if len(sents) == 1:
                arr.append(sents[0])
            else:
                arr.append(sents)
        f.close()
    return arr



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='USL-H training script')
    parser.add_argument('--metric', type=str, required=True, help='Choose a metric to train. VUP|NUP|MLM')
    parser.add_argument('--weight-path', type=str, default='./checkpoints', help='Path to directory that stores the weight')

    # Dataset
    parser.add_argument('--train_path', type=str, required=True, help='Path to the directory of training set')
    parser.add_argument('--valid_path', type=str, help='Path to the directory of validation set')
    parser.add_argument('--batch_size', type=int, default=16, help='samples per batches')
    parser.add_argument('--max-epochs', type=int, default=1, help='number of epoches to train')
    parser.add_argument('--num-workers', type=int, default=1, help='number of worker for dataset')
    parser.add_argument('--ctx-token-len', type=int, default=25, help='number of tokens for context')
    parser.add_argument('--res-token-len', type=int, default=25, help='number of tokens for response')

    # Modeling
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization')

    args = parser.parse_args()

    xtrain = read_dataset(args.train_path)
    xvalid = read_dataset(args.valid_path)
    print (xtrain)
    train(args, xtrain, xvalid)
