import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch
import random


def main(config):
    cudnn.benchmark = True
    # Check prefered model
    if config.model_type not in ['R2UNet','R2UNet_large']:
        print('ERROR!! model_type should be either R2UNet or R2UNet_large')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Check whether GPU is available or not
    if not torch.cuda.is_available():
        print('cuda is not available.')
        return
    else:
        print('cuda is available.')

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Printing out the model configuration
    print("config: ", config)
        
    # Get DataLoaders
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)
    print("---- before solver")
    solver = Solver(config, train_loader, valid_loader, test_loader)
    print("---- after solver")
    
    # Train and sample the images
    if config.mode == 'train':
        print("--- Model Summary --- ")
        solver.model_summary()
        print("---- Train starts")
        solver.train()
        print("---- Test starts")
        solver.test()
    elif config.mode == 'test':
        print("--- Model Summary --- ")
        solver.model_summary()
        print("---- Test starts")
        solver.test()
    elif config.mode == "model_summary":
        print("--- Model Summary --- ")
        solver.model_summary()
        solver.print_network()
    else:
        print('ERROR!! mode should be train, test or model_summary')
        print('Your input for mode was %s'%config.mode)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Default Hyperparameters
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--t', type=int, default=3, help='Time step for recurrent block')
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs_decay', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--beta1', type=float, default=0.5) # Adam optimizer parameter
    parser.add_argument('--beta2', type=float, default=0.999) # Adam optimizer parameter
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # Parameters for directorie, network and mode 
    parser.add_argument('--mode', type=str, default='train', help='train/test/model_summary') 
    parser.add_argument('--model_type', type=str, default='R2UNet', help='R2UNet/R2UNet_large')
    parser.add_argument('--model_path', type=str, default='../../models')
    parser.add_argument('--train_path', type=str, default='../../pet_dataset/train/')
    parser.add_argument('--valid_path', type=str, default='../../pet_dataset/valid/')
    parser.add_argument('--test_path', type=str, default='../../pet_dataset/test/')
    parser.add_argument('--result_path', type=str, default='../../result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
