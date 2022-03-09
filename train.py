import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import get_args
from model import Model
from dataset.dataset_wrapper import create_dataloaders

def train(model, train_loader, val_loader, args, device):
    
    ### Selecting Optimizer ###
    if(args.optim == "delta"):
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    elif(args.optim == "gen_delta"):
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
    elif(args.optim == "adam"):
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    else:
        raise Exception("Please provide a valid optimizer.")

    criterion = nn.CrossEntropyLoss()

    ### Training ###

    epoch_errors = [] # Average Error for every epoch

    for i in tqdm(range(args.num_epochs)):

        ### Weight Updates ###
        epoch_error = [] # Average Error for every batch
        for b, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)
            Y_pred = model(X_train)

            loss = criterion(Y_pred, Y_train)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            error = (torch.argmax(Y_pred, dim = 1) == Y_train).type(torch.float).sum().item()
            epoch_error.append(error)

        avg_epoch_error = sum(epoch_error)/len(epoch_error)
        epoch_errors.append(avg_epoch_error)

        ### Save Weights ###
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'epoch_{i+1}.pt'))

    plt.plot(epoch_errors)
    plt.show()


def main():
    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup model
    model = Model(args)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Save/load initial weights
    if(args.save_init):
        torch.save(model.state_dict(), args.init_path)
    else:
        model.load_state_dict(torch.load(args.init_path))

    train_loader, val_loader, _ = create_dataloaders(args)
    train(model, train_loader, val_loader, args, device)

if __name__=="__main__":
    main()