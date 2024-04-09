# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options: 
# * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
# * Choose architecture: python train.py data_dir --arch "vgg13" 
# * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# * Use GPU for training: python train.py data_dir --gpu

from utils import init_data, set_device, model_setup, train_model, save_checkpoint
import argparse

arg_parser = argparse.ArgumentParser(description='train.py')

arg_parser.add_argument('data_dir', nargs = '*', action = "store", default = "flowers")
arg_parser.add_argument('--save_dir', dest = "save_dir", action = "store", default = "./")
arg_parser.add_argument('--arch', dest = "arch", action = "store", default = "vgg16", type = str)
arg_parser.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.001)
arg_parser.add_argument('--hidden_units', dest = "hidden_units", action = "store", default = 4096, type = int)
arg_parser.add_argument('--epochs', dest = "epochs", action = "store", default = 5, type = int)
arg_parser.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu", type = str)

params = arg_parser.parse_args()

device = set_device(params.gpu)

image_datasets, data_loaders = init_data(params.data_dir)

model, optimizer, criterion = model_setup(device, params.arch, params.hidden_units, params.learning_rate)
print(model)
            
train_model(data_loaders['train_data_loader'], data_loaders['validation_data_loader'], image_datasets['train_data'], model, optimizer, criterion, device, params.epochs)

save_checkpoint(model, params.save_dir)