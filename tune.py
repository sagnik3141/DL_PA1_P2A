import os
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config_tune import get_args
from model import Model
from dataset.dataset_wrapper import create_dataloaders
from display_util import plot_errors
from train import train

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

    ### Tuning ###
    tune_list = []
    for hparam_trial in args.trials(30):
        val_error = train(model, train_loader, val_loader, hparam_trial, device, tune = True)
        tune_list.append((val_error, hparam_trial))
    with open('tune_list.pkl', 'wb') as fp:
        pickle.dump(tune_list, fp)

if __name__=="__main__":
    main()