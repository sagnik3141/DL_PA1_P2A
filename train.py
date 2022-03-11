import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import get_args
from model import Model
from dataset.dataset_wrapper import create_dataloaders
from display_util import plot_errors

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

    writer = SummaryWriter('runs/train') # Tensorboard for plots
    epoch_errors = []
    val_epoch_errors = []
    for i in tqdm(range(args.num_epochs)):

        ### Weight Updates ###
        epoch_error = [] # Average Error for every batch
        epoch_loss = 0
        for b, (X_train, Y_train) in enumerate(train_loader):
            X_train = X_train.to(device).float()
            Y_train = Y_train.type(torch.LongTensor).to(device)
            Y_pred = model(X_train)

            loss = criterion(Y_pred, Y_train)
            epoch_loss+=loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            error = (torch.argmax(Y_pred, dim = 1) != Y_train).type(torch.float).sum().item()
            epoch_error.append(error)

        avg_epoch_error = sum(epoch_error)/len(epoch_error)
        epoch_errors.append(avg_epoch_error)

        ### Validation ###
        val_errors = []
        val_loss = 0
        for b, (X_val, Y_val) in enumerate(val_loader):
            X_val = X_val.to(device).float()
            Y_val = Y_val.type(torch.LongTensor).to(device)
            Y_pred = model(X_val)

            loss = criterion(Y_pred, Y_val)
            val_loss+=loss
            
            error = (torch.argmax(Y_pred, dim = 1) != Y_val).type(torch.float).sum().item()
            val_errors.append(error)

        avg_val_error = sum(val_errors)/len(val_errors)
        val_epoch_errors.append(avg_val_error)

        # Plotting
        writer.add_scalars('Loss', {
            'Train': epoch_loss,
            'Validation': val_loss,
        }, i+1)

        writer.add_scalars('Error', {
            'Train': avg_epoch_error,
            'Validation': avg_val_error,
        }, i+1)
        ### Save Weights ###
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f'epoch_{i+1}.pt'))

    # Error Plots
    plot_errors(epoch_errors, val_epoch_errors)


def main():
    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup model
    model = Model(args)
    #model = nn.DataParallel(model)
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