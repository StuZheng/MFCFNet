import os
import sys
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from DataSet import MyDataSet
import csv 
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
#from AlexNet_Multi import AlexNet_Multi as alexNet
#from vgg16_Multi import vgg11 as vgg11
#from vgg16_Multi import vgg11_bn as vgg11_bn
#from vgg16_Multi import vgg16 as vgg16
#from vgg16_Multi import vgg16_bn as vgg16_bn
#from vgg16_Multi import vgg19 as vgg19
#from vgg16_Multi import vgg19_bn as vgg19_bn

#from resnet18_Multi import resnet18 as resnet18
#from resnet18_Multi import resnet101 as resnet101
#from resnet18_Multi import resnet152 as resnet152

from densenet121_MultiCNN import densenet121 as densenet121
from densenet121_MultiCNN import densenet169 as densenet169
from densenet121_MultiCNN import densenet201 as densenet201

#from squeezenet_Multi import squeezenet1_0 as squeezenet1_0
#from ConvNeXt_Multi import convnext_tiny
#from ConvNeXt_Multi import convnext_small
#from ConvNeXt_Multi import convnext_base
#from ConvNeXt_Multi import convnext_large
#from ConvNeXt_Multi import convnext_xlarge


from PCA_tool import *
#from model import resnet34
# Ori_Models
from torchvision.models.densenet import densenet121 as M_densenet121
from torchvision.models.densenet import densenet169 as M_densenet169
from torchvision.models.densenet import densenet201 as M_densenet201
from torchvision.models.alexnet import alexnet as M_alexnet
from torchvision.models.vgg import vgg11 as M_vgg11
from torchvision.models.vgg import vgg16 as M_vgg16
from torchvision.models.vgg import vgg16_bn as M_vgg16_bn
from torchvision.models.vgg import vgg19 as M_vgg19
from torchvision.models.vgg import vgg19_bn as M_vgg19_bn
from torchvision.models.resnet import resnet18 as M_resnet18
from torchvision.models.resnet import resnet101 as M_resnet101
from torchvision.models.resnet import resnet152 as M_resnet152

from torchvision.models.squeezenet import SqueezeNet as M_SqueezeNet

# from torchvision.models.mobilenet import mobilenet_v2
from torch.optim.lr_scheduler import StepLR,MultiStepLR,CyclicLR
from skimage.feature import hog, local_binary_pattern, corner_harris, corner_peaks, canny
from skimage import color
from skimage import io
from skimage import filters
import cv2
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tool import EarlyStopping
from tool_getMeanAStd import *



# seed
def _init_fn(seed=3407):
    # np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def getDataSet(mode,dataPath):
    csvFile = open(dataPath, "r")
    reader = csv.reader(csvFile)
    data= []
    label = []
    for item in reader:
        if reader.line_num == 1:
                continue
        data.append(item[0])
        label.append(int(item[1]))
    csvFile.close()
    data = np.array(data)
    label = np.array(label)
    dataSet = MyDataSet(mode,data,label)
    return dataSet

# Hand Feature
def extractHOG(images):   

    train_data_vector=[]
    for image in images:
        normalised_blocks= hog(image.numpy().squeeze(), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',channel_axis=0, visualize=False)    
        normalised_blocks = Normalize_tool(normalised_blocks) 
        train_data_vector.append(normalised_blocks)
    
    train_data_vector = torch.Tensor(np.array(train_data_vector))
    return train_data_vector

def extractLBP(images):

    train_data_vector=[]
    for image in images:
        image = color.rgb2gray(np.transpose(image,(1,2,0)))
        lbp = local_binary_pattern(image, 24,8, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, 27),range=(0, 26)) 
        hist = hist.astype(np.float32) 
        hist /= (hist.sum() + (1e-7))
        train_data_vector.append(hist)
 
    train_data_vector = torch.Tensor(np.array(train_data_vector))
    return train_data_vector

def extractLBP_ori(images):
   
    train_data_vector=[]
    for image in images:
        image = color.rgb2gray(np.transpose(image,(1,2,0)))
        lbp = local_binary_pattern(image, 8,3, method="uniform")
        lbp = lbp.ravel()
        lbp = lbp.astype(np.float32) 
        lbp = Normalize_tool(lbp)  
        train_data_vector.append(lbp)

    train_data_vector = np.array(train_data_vector)
    train_data_vector = torch.Tensor(train_data_vector)
    return train_data_vector

def extractHarris_ori(images):

    train_data_vector=[]
    for image in images:
        image = color.rgb2gray(np.transpose(image,(1,2,0)))
        harris = corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1)
        harris = harris.ravel()
        harris = harris.astype(np.float32) 
        harris = Normalize_tool(harris) 
        train_data_vector.append(harris)
    train_data_vector = torch.Tensor(np.array(train_data_vector))
    return train_data_vector

def extractHarris(images):

    train_data_vector=[]
    for image in images:
        image = color.rgb2gray(np.transpose(image,(1,2,0)))
        harris = corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1)
        harris = corner_peaks(harris, min_distance=1)
        max_bins = int(harris.max() + 1)
        (harris, _) = np.histogram(harris.ravel(), bins=512,range=(0, max_bins))  
        harris = harris.astype(np.float32) 
        harris /= (harris.sum() + (1e-7))
        print(harris.shape)
        train_data_vector.append(harris)


    train_data_vector = torch.Tensor(np.array(train_data_vector))
    return train_data_vector

def extractSIFT(images):
    train_data_vector=[]
    for image in images:
       
        gray_image = cv2.cvtColor(np.asarray(np.transpose(image,(1,2,0))), cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype(np.uint8)
        sift = cv2.SIFT_create() 
        step_size = 12
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray_image.shape[0],step_size)
                                            for x in range(0, gray_image.shape[1],step_size)]
        dense_feat, desc = sift.compute(gray_image, kp)
        if desc is not None:     
            desc = desc.flatten()
            desc = desc.astype(np.float32) 
            desc /= (desc.sum() + (1e-7))
            train_data_vector.append(desc)

    train_data_vector = torch.Tensor(np.array(train_data_vector))
    
    return train_data_vector


def extractGabor(images):
    train_data_vector=[]
    for image in images:
        gray_image = cv2.cvtColor(np.asarray(np.transpose(image,(1,2,0))), cv2.COLOR_BGR2GRAY) 
        gray_image = gray_image.astype(np.uint8)

        #gabor 
        real, imag = filters.gabor(gray_image, frequency=0.6,theta=45,n_stds=5)

        img_mod=np.sqrt(real.astype(float)**2+imag.astype(float)**2)
        #
        newimg = cv2.resize(img_mod,(0,0),fx=1,fy=1,interpolation=cv2.INTER_AREA)
        tempfea = newimg.flatten()

        train_data_vector.append(tempfea)
    
    train_data_vector = torch.Tensor(np.array(train_data_vector))
    
    return train_data_vector


def extractCanny(images):
    train_data_vector=[]
    for image in images:
        gray_image = cv2.cvtColor(np.asarray(np.transpose(image,(1,2,0))), cv2.COLOR_BGR2GRAY) 
       
        canny_edge = canny(gray_image, sigma=3)
        tempfea = canny_edge.flatten()
        
        train_data_vector.append(tempfea)

    
    train_data_vector = torch.Tensor(np.array(train_data_vector))
    
    return train_data_vector


def imageToGary(images):
    train_data_vector=[]
    for image in images:
        gray_image = color.rgb2gray(np.transpose(image,(1,2,0)))
        gray_image = np.expand_dims(gray_image,axis=0)  
        train_data_vector.append(gray_image)

    train_data_vector = torch.Tensor(np.array(train_data_vector))    
    return train_data_vector

def main():

    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print("Start Tensorboard with 'tensorboard  --logdir=runs', view at http://localhost:6006/")
    # SummaryWriter
    timestr = time.strftime('%Y%m%d_%H')
    tb_writer = SummaryWriter(log_dir="runs/FUSARship_experiment/"+timestr)

    # 1.Prepare dataset
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) 
    print('Using {} dataloader workers every process'.format(nw))
    
	# openSAR
    train_dataset = getDataSet("train","/home/xxx/workspace/data/SARShip/openSARShip/train.csv") 
    val_dataset = getDataSet("test","/home/xxx/workspace/data/SARShip/openSARShip/val.csv") 

    # FUSAR_v2版数据集
    # train_dataset = getDataSet("train","/home/xxx/workspace/data/SARShip/FUSAR_v2/train.csv") #([0.0375913, 0.0375913, 0.0375913], [0.05455397, 0.05455397, 0.05455397])
    # val_dataset = getDataSet("test","/home/xxx/workspace/data/SARShip/FUSAR_v2/val.csv")   #([0.035997953, 0.035997953, 0.035997953], [0.053638346, 0.053638346, 0.053638346])
    
    

    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size = batch_size,
                            num_workers=nw,
                            drop_last=True)

    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            batch_size = batch_size,
                            num_workers=nw,
                            drop_last=True)
   
    # set seed
    _init_fn()

    train_num = len(train_dataset)
    val_num   = len(val_dataset)

    
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # MFCFNet
    # net = resnet18(num_classes=7)
    # net = resnet101(num_classes=7)
    # net = resnet152(num_classes=7)
    # net = vgg16(num_classes=3)
    # net = vgg16_bn(num_classes=7)
    # net = vgg19(num_classes=3)
    # net = vgg11(num_classes=7)
    # net = vgg11_bn(num_classes=7)
    # net = vgg19_bn(num_classes=3)
    # net = alexNet(num_classes=3)
    net = densenet121(num_classes=3)
    # net = densenet169(num_classes=3)
    # net = densenet201(num_classes=3)
    # net = squeezenet1_0(num_classes=3)
    
	# Ori_Net
    # net = M_alexnet(num_classes=3)
    # net = M_resnet18(num_classes=3)
    # net = M_resnet101(num_classes=7)
    # net = M_resnet152(num_classes=7)
    # net = M_vgg11(num_classes=3)
    # net = M_vgg16(num_classes=3)
    # net = M_vgg16_bn(num_classes=7)
    # net = M_vgg19(num_classes=3)
    # net = M_vgg19_bn(num_classes=3)
    # net = M_densenet121(num_classes=3)
    # net = M_densenet169(num_classes=3)
    # net = M_densenet201(num_classes=3)
    # net = M_SqueezeNet(num_classes=3)
    # net = convnext_tiny(num_classes=7)
    
    
    net.to(device)

    init_img= torch.zeros((1,3,224,224), device=device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5) 
  
    scheduler_1 = StepLR(optimizer, step_size=60, gamma=0.1)



    epochs = 100
    best_acc = 0.0
    overfit_acc = 0.0
    overfit_epoch = 0
    val_loss_low = 5.0
    best_confusion=0
   
    save_path = 'xxx.pth'
    early_stopping = EarlyStopping(patience=20, verbose=False)  # 早停
    
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            
            # extract hand feature and inject
            
            train_hand_2D_1 = extractGabor(images)           

            logits,loss_0 = net(images.to(device),train_hand_2D_1.to(device),labels.to(device))
            
            loss = loss_function(logits, labels.to(device))            
            loss = loss + loss_0
            
            loss.backward()
            optimizer.step()
        
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        scheduler_1.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_loss = 0.0
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
            
                val_hand_2D_1 = extractGabor(val_images)


                outputs,_ = net(val_images.to(device),val_hand_2D_1.to(device),val_labels.to(device))

                predict_y = torch.max(outputs, dim=1)[1]

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                loss_val = loss_function(outputs, val_labels.to(device))

                val_loss += loss_val.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)


        val_accurate = acc / val_num
     
        # add loss, acc, and lr into tensorboard
        tags = ["train_loss", "val_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
        tb_writer.add_scalar(tags[1], val_loss / val_steps, epoch)
        tb_writer.add_scalar(tags[2], val_accurate, epoch)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)


        if (val_loss / val_steps) < val_loss_low:
            val_loss_low = (val_loss / val_steps)
            if val_accurate > best_acc:
                best_acc = val_accurate
                # torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f val_accuracy: %.4f' % 
              (epoch + 1, running_loss / train_steps, val_loss / val_steps, val_accurate))
    tb_writer.close()
    print('Finished Training, the best accuracy is %.4f, the lowest loss is %.4f' % (best_acc, val_loss_low))
    
   

if __name__ == '__main__':
    main()
    


