import argparse

def get_args():
    
    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument('--input_dim', type = int, default = 60, help = "Input dimensions of data")
    parser.add_argument('--num_classes', type = int, default = 5, help = "Number of output classes.")
    parser.add_argument('--num_nodes_h1', type = int, default = 30, help = "Number of nodes in hidden layer 1.")
    parser.add_argument('--num_nodes_h2', type = int, default = 10, help = "Number of nodes in hidden layer 2.")

    # Dataset Args
    parser.add_argument('--data_path', type = str, default = './data/single_label_image_dataset/image_data_dim60.txt')
    parser.add_argument('--labels_path', type = str, default = './data/single_label_image_dataset/image_data_labels.txt')
    parser.add_argument('--val_split',  type = float, default = 0.15, help = "Fraction of validation data.")
    parser.add_argument('--test_split',  type = float, default = 0.15, help = "Fraction of test data.")
    parser.add_argument('--shuffle_data', type = bool, default = True, help = "If True, shuffle dataset.")
    parser.add_argument('--random_seed', type = int, default = 60)
    parser.add_argument('--batch_size', type = int, default = 1)

    # Train Args
    parser.add_argument('--save_init', type = bool, default = False, help = "Save initial weights for later experiments.")
    parser.add_argument('--ckpt_dir', type = str, default = './checkpoints', help = "Directory for saving model weights.")
    parser.add_argument('--init_path', type = str, default = './checkpoints/epoch_init.pt', help = "Path to initial random weights.")
    parser.add_argument('--optim', type = str, default = 'delta', help = "Optimizer to use. Options: delta (SGD)| gen_delta (SGD with momentum) | adam (Adam)")
    parser.add_argument('--lr', type = float, default = 1e-4, help = "Learning Rate")
    parser.add_argument('--momentum', type = float, default = 0.9, help = "Optimizer Momentum")
    parser.add_argument('--num_epochs', type = int, default = 250, help = "Total number of epochs")

    # Test Args
    parser.add_argument('--ckpt_best_path', type = str, default = './checkpoints/epoch_200.pt', help = "Path to weights used for testing.")
    
    args = parser.parse_args()

    return args