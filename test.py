import os
import torch
import torch.nn as nn

from config import get_args
from model import Model
from dataset.dataset_wrapper import create_dataloaders
from display_util import plot_confusion_matrix

def test(model, test_loader, args, device):
    
    criterion = nn.CrossEntropyLoss()
    test_errors = []
    test_loss = 0
    true_labels = []
    pred_labels = []

    for b, (X_test, Y_test) in enumerate(test_loader):
        X_test = X_test.to(device).float()
        Y_test = Y_test.type(torch.LongTensor).to(device)
        Y_pred = model(X_test)

        loss = criterion(Y_pred, Y_test)
        test_loss+=loss
        
        error = (torch.argmax(Y_pred, dim = 1) != Y_test).type(torch.float).sum().item()
        test_errors.append(error)
        true_labels.append(Y_test.item())
        pred_labels.append(torch.argmax(Y_pred, dim = 1).item())

    avg_val_error = sum(test_errors)/len(test_errors)
    
    print('Average error: {:.4f}'.format(avg_val_error))
    
    plot_confusion_matrix(true_labels, pred_labels)

def main():
    
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(args)
    model = model.to(device)
    model.load_state_dict(torch.load(args.ckpt_best_path))

    _, __, test_loader = create_dataloaders(args)
    test(model, test_loader, args, device)


if __name__=="__main__":
    main()