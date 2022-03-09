import os
import argparse
from tqdm import tqdm
import torch

from config import get_args
from model import Model
from dataset.dataset_wrapper import create_dataloaders

def train(model, train_loader, val_loader, args):
    
    ### Selecting Optimizer ###
    if(args.optim == "delta"):
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    elif(args.optim == "gen_delta"):
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
    elif(args.optim == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    else:
        raise Exception("Please provide a valid optimizer.")

    ### Training ###

    for i in tqdm(range(args.num_epochs)):
        pass


def main():
    
    args = get_args()
    
    model = Model(args)
    #model = nn.DataParallel(model)

    train_loader, val_loader, _ = create_dataloaders(args)
    train(model, train_loader, val_loader, args)

if __name__=="__main__":
    main()