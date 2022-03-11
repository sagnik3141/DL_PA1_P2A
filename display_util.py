import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_errors(train_errors, val_errors):
    """
    This function generates error plots.
    """

    plt.plot(train_errors)
    plt.plot(val_errors)
    plt.legend(['Training Error', 'Validation Error'])
    plt.xlabel(r'Epochs $\rightarrow$')
    plt.ylabel(r'Error $\rightarrow$')
    plt.title(r'Error Plot')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels):
    """
    This function plots the confusion matrix.
    """

    matrix = confusion_matrix(true_labels, pred_labels)
    print(matrix)