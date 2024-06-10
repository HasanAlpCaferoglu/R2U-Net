import os
import json
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import R2UNet_large, R2UNet
import csv
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None # model is set in build_model function
        self.optimizer = None # optimizer is set in build_model funtion
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyperparameters
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.t = config.t
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path variables
        self.model_path = config.model_path
        self.result_path = config.result_path
        
        # Mode
        self.mode = config.mode

        # Device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            
            gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7] # IDs of all available GPUs
            gpu_id = 1
            torch.cuda.set_device(gpu_ids[gpu_id])
            device = torch.device('cuda:' + str(gpu_ids[gpu_id]))

            print(f'Device: {device}')
        else:
            raise RuntimeError("CUDA (GPU) is not available. Make sure you have installed the necessary drivers and libraries.")
        
        # Build the model
        self.model_type = config.model_type
        self.build_model()

    def build_model(self):
        """
        Construct model
        """
        # initializing model instance
        if self.model_type == "R2UNet":
            self.unet = R2UNet(img_ch=3,output_ch=self.output_ch, t=self.t)
        elif self.model_type == "R2UNet_large":
            self.unet = R2UNet_large(img_ch=3,output_ch=self.output_ch , t=self.t)
        else:
            raise RuntimeError("Please select appropriate model. You can select either R2UNet or R2UNet_large")
        
        # Setting Adam optimizer as optimizer
        self.optimizer = optim.Adam(list(self.unet.parameter()))


    def model_summary(self):
        """
        Prints the model summary and the total number of parameters
        """
        print('Model summary is given below:')
        total_params = sum(p.numel() for p in list(self.unet.parameters()))
        print("Total number of parameters: ", total_params)
        return
    
    def print_network(self, model, name):
        """
        Print out the network information.
        """
        num_params = 0
        for p in list(model.parameters()):
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    
    def reset_grad(self):
        """
        Zero out the gradient buffers.
        """
        self.optimizer.zero_grad()

    def save_img_to_dir(self, sr_img, gt_img, filename, path='../../example_images_test/'):
        """"
        This function saves both predicted/segmented image and grount truth image in the provided directory.
        This function is used for extracting exampled during and after training and compare predicted/segmented
        image with its ground truth.
        """
        sr_img = sr_img.cpu()
        plt.imshow(sr_img, cmap='gray')
        plt.title('sr_img')
        plt.show()
        plt.savefig(path + filename + '_SR_' + '.png')

        gt_img = torch.squeeze(gt_img, dim=0).cpu()
        plt.imshow(gt_img, cmap='gray')
        plt.title('gt_img')
        plt.show()
        plt.savefig(path  + filename + '_GT_' + '.png')	
        return

    def closing(self, img_matrix):
        """
        This is a morphological operation constructed with two morphological operation, 
        starting with dilation and then applying erosion.
        """
        kernel = np.ones((5,5), np.uint8)
        img_dilated = cv2.dilate(img_matrix, kernel, iterations=1) 
        img_closed = cv2.erode(img_dilated, kernel, iterations=1)
        return img_closed
    
    def opening(self, img_matrix):
        """
        This is a morphological operation constructed with two morphological operation, 
        starting with erosion and then applying dilation.
        """    
        kernel = np.ones((5,5), np.uint8)
        img_eroded = cv2.erode(img_matrix, kernel, iterations=1)
        img_opened = cv2.dilate(img_eroded, kernel, iterations=1) 
        return img_opened

    def dilation(self, img_matrix):
        """"
        The function applied dilation morphological operation on the given image matrix
        """
        kernel = np.ones((5,5), np.uint8)
        img_dilated = cv2.dilate(img_matrix, kernel, iterations=1) 
        return img_dilated

    
    def erosion(self, img_matrix):
        """
        The function applied erosion morphological operation on the given image matrix
        """
        kernel = np.ones((5,5), np.uint8)
        img_eroted = cv2.erode(img_matrix, kernel, iterations=1) 
        return img_eroted


    def train(self):
        """"
        This is a function for training model.
        This function also includes validation part.
        """
        #====================================== Training ===========================================#
        print("--- Train started...")
        # defining a path for a specific model with its hyperparameters
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f-%dbs-%dt.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob, self.batch_size, self.t))

        # If the model exist, load it 
        if os.path.isfile(unet_path):
            print("Model path (unet_path): ", unet_path)
            # Load the pretrained model
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is successfully loaded from %s'%(self.model_type, unet_path))
        else:
            # Training 
            lr = self.lr
            best_unet_score = 0.

            all_epoch_train_metric_info = []
            all_valid_metric_info = []
            for epoch in range(self.epochs):
                print("Epoch: ", epoch)
                torch.cuda.empty_cache()
                self.unet.train(True) # making train mode open
                epoch_loss = 0.
                epoch_loss_validation = 0.

                # setting measurement metrics
                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length = 0

                for i, (images, GT, filenames) in enumerate(self.train_loader):
                    # Pixel mapping for the ground truth image
                    # This part is valid for the Oxford PET-III Dataset
                    # With pixel mapping, the borders of the animals are removed
                    GT[GT < 0.0070] = 1
                    GT[GT > 0.0080] = 1
                    GT[GT!=1] = 0
                    # moving vectors to device
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    # squeezing the ground truth so that it becomes 1D
                    GT = torch.squeeze(GT, dim=1)
                    GT = GT.long()
                    # segmenting
                    SR = self.unet(images)
                    # finding loss 
                    loss = self.criterion(SR, GT)
                    epoch_loss += loss.item()
                    # backprop + optimeze
                    self.optimizer.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    # calculating evaluation metrics for the current batch  
                    SR = F.sigmoid(SR)
                    SR = torch.argmax(SR, dim=1) # to get the classes
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)

                    length += 1
                
                # calculating evaluation metrics for the current epoch for training
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                # Decay learning rate
                if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print ('Decay learning rate to lr: {}.'.format(lr))

                epoch_metric_info = {'mode': 'train', 'epoch': epoch, 'loss:': epoch_loss, 'acc': acc, 'SE': SE, 'SP': SP, 'PC': PC, 'F1':F1, 'JS': JS, 'DC': DC}
                all_epoch_train_metric_info.append(epoch_metric_info)

                # Print the epoch result
                print('Epoch [%d/%d], \n[Training] Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch+1, self.num_epochs, \
                        epoch_loss,\
                        acc,SE,SP,PC,F1,JS,DC))
                    
                #===================================== Validation ====================================#
                """
                Extacting the validation metrics for the current epoch
                """
                self.unet.train(False)
                self.unet.eval()
                # setting evaluation metrics to zero for the validation
                acc = 0.	# Accuracy
                SE = 0.		# Sensitivity (Recall)
                SP = 0.		# Specificity
                PC = 0. 	# Precision
                F1 = 0.		# F1 Score
                JS = 0.		# Jaccard Similarity
                DC = 0.		# Dice Coefficient
                length = 0
                
                for i,(images, GT, filenames) in enumerate(self.valid_loader):
                    with torch.no_grad():
                        images = images.to(self.device)
                        # Pixel mapping for the ground truth image
                        # This part is valid for the Oxford PET-III Dataset
                        # With pixel mapping, the borders of the animals are removed
                        GT[GT < 0.0070] = 1
                        GT[GT > 0.0080] = 1
                        GT[GT != 1] = 0
                        # loading images to the device
                        GT = GT.to(self.device)
                        # squeezing the ground truth so that it becomes 1D
                        GT = torch.squeeze(GT, dim=1)
                        GT = GT.long()

                        SR = self.unet(images)
                        SR_for_loss = SR
                        GT_for_loss = GT
                        loss_in_valid = self.criterion(SR_for_loss, GT_for_loss)
                        epoch_loss_validation += loss_in_valid.item()
                        SR = F.sigmoid(SR)
                        SR = torch.argmax(SR, dim=1) # to get the classes
                        acc += get_accuracy(SR,GT)
                        SE += get_sensitivity(SR,GT)
                        SP += get_specificity(SR,GT)
                        PC += get_precision(SR,GT)
                        F1 += get_F1(SR,GT)
                        JS += get_JS(SR,GT)
                        DC += get_DC(SR,GT)
                            
                        length += 1
                
                # calculating evaluation metrics for the current epoch for validation
                acc = acc/length
                SE = SE/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                JS = JS/length
                DC = DC/length

                print('[Validation] Loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(epoch_loss_validation,acc,SE,SP,PC,F1,JS,DC))
                all_valid_metric_info.append({ 'mode': 'valid','epoch': epoch, 'loss': epoch_loss_validation, 'acc': acc, 'SE': SE, 'SP': SP, 'PC': PC, 'F1':F1, 'JS': JS, 'DC': DC})

                # calculating unet score based on the validation set
                unet_score = JS + DC

                # Saving Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(best_unet,unet_path)
            
            #===================================== END OF TRAINING ====================================#
            # Collect image examaples from validation set
            # From each batch collect 4 images
            for i, (images, GT, filenames) in enumerate(self.valid_loader):
                images = images.to(self.device)
                GT[GT < 0.0070] = 1
                GT[GT > 0.0080] = 1
                GT[GT != 1] = 0
                GT = GT.to(self.device)
                GT = torch.squeeze(GT, dim=1)
                GT = GT.long()

                self.unet.train(False)
                SR = self.unet(images)
                SR = F.sigmoid(SR)
                SR = torch.argmax(SR, dim=1) # to get the classes
                
                for k in range(min(4, self.batch_size)):
                    SR_k = SR[k].cpu().numpy()
                    plt.imshow(SR_k, cmap='gray')
                    plt.title('SR')
                    plt.show()

                    saved_img_dir = '../../' + str(self.model_type) + '_example_images_' + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob) + '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't'+ "/"
                    # Check if the directory exists, create it if it doesn't
                    if not os.path.exists(saved_img_dir):
                        os.makedirs(saved_img_dir)

                    plt.savefig(saved_img_dir + filenames[k] + '_SR_'+ str(i) + '_' + str(k) + '.png')

                    GT_k = torch.squeeze(GT[k], dim=0).cpu().numpy()
                    plt.imshow(GT_k, cmap='gray')
                    plt.title('GT')
                    plt.show()
                    plt.savefig(saved_img_dir + filenames[k] + '_GT_' + str(i) + '_' + str(k) + '.png')	

                # After collection of the 4 image from the batch, break the loop
                if i == 4:
                    break

            print('all_epoch_train_metric_info: ', all_epoch_train_metric_info)
            print('all_valid_metric_info: ', all_valid_metric_info)
            file_path_train_metrics = '../../train_and_valid_metric_logs/train_metric_info_' + str(self.model_type) + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob) + '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' +  '.json'
            file_path_valid_metrics = '../../train_and_valid_metric_logs/valid_metric_info_' + str(self.model_type) + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob) + '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' + '.json'
            with open(file_path_train_metrics, 'w') as json_file:
                json.dump(all_epoch_train_metric_info, json_file)
            with open(file_path_valid_metrics, 'w') as json_file:
                json.dump(all_valid_metric_info, json_file)

            # Write results
            f = open(os.path.join(self.result_path,'result.csv'), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob])
            f.close()


            
    def test(self):
        print('--- Test ---')
        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f-%dbs-%dt.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob, self.batch_size, self.t))
        
        # check if there is a model file
        if os.path.isfile(unet_path):
            # Loading the model
            self.unet.load_state_dict(torch.load(unet_path))
            print("%s is successfully loaded from %s"%(self.model_type, unet_path))

            self.unet.train(False)
            self.unet.eval()

            # setting evaluation metrics to 0
            acc = 0.    # accuracy
            SE = 0.		# Sensitivity (Recall)
            SP = 0.		# Specificity
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            JS = 0.		# Jaccard Similarity
            DC = 0.		# Dice Coefficient
            length = 0
            test_batches_metric_info = []
            # iterating over test data
            for batch_no, (images, GT, filenames) in enumerate(self.test_loader):
                with torch.no_grad():
                    print("-- batch ", batch_no)
                    images = images.to(self.device)
                    # pixel mapping
                    GT[GT < 0.0070] = 1
                    GT[GT > 0.0080] = 1
                    GT[GT != 1] = 0
                    GT = GT.to(self.device)
                    GT = torch.squeeze(GT, dim=1)
                    GT = GT.long()
                    # inference
                    SR = self.unet(images)
                    SR = F.sigmoid(SR)
                    SR = torch.argmax(SR, dim=1) # to get the classes

                    # updating evaluation metrics
                    acc += get_accuracy(SR,GT)
                    SE += get_sensitivity(SR,GT)
                    SP += get_specificity(SR,GT)
                    PC += get_precision(SR,GT)
                    F1 += get_F1(SR,GT)
                    JS += get_JS(SR,GT)
                    DC += get_DC(SR,GT)

                    length += 1

                    test_batches_metric_info.append({
                        'mode':'test', 
                        'batch': batch_no, 
                        'acc': get_accuracy(SR,GT), 
                        'SE': get_sensitivity(SR,GT), 
                        'SP': get_specificity(SR,GT), 
                        'PC': get_precision(SR,GT), 
                        'F1':get_F1(SR,GT), 
                        'JS': get_JS(SR,GT), 
                        'DC': get_DC(SR,GT),
                        'filename_sample': [fname for fname in filenames],
                    })

                    # Observe fails
                    # if DC is less then 0.1 save images to observe this batch
                    if get_DC(SR, GT) < 0.1:
                        failed_prediction_img_dir = '../../'  + str(self.model_type) + '_FAIL_imgs' + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob)+ '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' + "_test/"
                        # Check if the directory exists, create it if it doesn't
                        if not os.path.exists(saved_img_dir):
                            os.makedirs(failed_prediction_img_dir)
                        self.save_img_to_dir(sr_img=SR[0], gt_img=GT[0], filename=filenames[1]+'_batch_'+str(batch_no), path=failed_prediction_img_dir)
                        self.save_img_to_dir(sr_img=SR[1], gt_img=GT[1], filename=filenames[1]+'_batch_'+str(batch_no), path=failed_prediction_img_dir)
                        
                    saved_img_dir = '../../'  + str(self.model_type) + '_example_images_' + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob)+ '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' + "_test/"
                    # Check if the directory exists, create it if it doesn't
                    if not os.path.exists(saved_img_dir):
                        os.makedirs(saved_img_dir)
                    self.save_img_to_dir(sr_img=SR[1], gt_img=GT[1], filename=filenames[1]+'_batch_'+str(batch_no), path=saved_img_dir)

            # Calculating whole test set metrics
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            unet_score = JS + DC

            print('[Test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
            test_metric_info = {
                'mode': 'test',
                'acc': acc, 
                'SE': SE, 
                'SP': SP, 
                'PC': PC, 
                'F1':F1, 
                'JS': JS, 
                'DC': DC
            }
            
            file_path_test_metrics = '../../test_metric_logs/test_metric_info_' + str(self.model_type)  + str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob) + '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' + '.json'
            file_path_test_batch_metrics = '../../test_metric_logs/test_metric_batch_info_' + str(self.model_type) +  str(self.num_epochs)+ '_' + str(self.lr) + '_' + str(self.num_epochs_decay) + '_' + str(self.augmentation_prob) + '_' + str(self.batch_size) + 'bs_' + str(self.t) + 't' + '.json'

            with open(file_path_test_metrics, 'w') as json_file:
                json.dump(test_metric_info, json_file)
            with open(file_path_test_batch_metrics, 'w') as json_file:
                json.dump(test_batches_metric_info, json_file)
        else:
            print("There is no model file")
