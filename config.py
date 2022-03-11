import argparse

def get_args():
    
    parser = argparse.ArgumentParser()

    # Model Args
    parser.add_argument('--input_dim', type = int, default = 60)
    parser.add_argument('--num_classes', type = int, default = 5)
    parser.add_argument('--num_nodes_h1', type = int, default = 30)
    parser.add_argument('--num_nodes_h2', type = int, default = 10)

    # Dataset Args
    parser.add_argument('--data_path', type = str, default = './data/single_label_image_dataset/image_data_dim60.txt')
    parser.add_argument('--labels_path', type = str, default = './data/single_label_image_dataset/image_data_labels.txt')
    parser.add_argument('--val_split',  type = float, default = 0.15)
    parser.add_argument('--test_split',  type = float, default = 0.15)
    parser.add_argument('--shuffle_data', type = bool, default = True)
    parser.add_argument('--random_seed', type = int, default = 60)
    parser.add_argument('--batch_size', type = int, default = 1)

    # Train Args
    parser.add_argument('--save_init', type = bool, default = False)
    parser.add_argument('--ckpt_dir', type = str, default = './checkpoints')
    parser.add_argument('--init_path', type = str, default = './checkpoints/epoch_init.pt')
    parser.add_argument('--optim', type = str, default = 'delta')
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--num_epochs', type = int, default = 250)

    # Test Args
    parser.add_argument('--ckpt_best_path', type = str, default = './checkpoints/epoch_200.pt')
    
    args = parser.parse_args()

    return args