from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os,sys

import argparse

import matplotlib.pyplot as plt
import pylab as pl



from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

import mrcfile
import random
from tqdm import tqdm

import csv
from sklearn.metrics import f1_score

class CreateDatasetFromImages(Dataset):
    def __init__(self, csv_path, file_path, transform_):
        """
        Args:
            csv_path (string): csv file
            img_path (string): image file
            transform: transform 
        """
        
        #self.resize_height = resize_height
        #self.resize_width = resize_width
 
        # csv_path = "C:UsersandroidcatDesktopcancer_classificationWarwick QU Dataset (Released 2016_07_08)Grade_train.csv"
        self.file_path = file_path
        self.to_tensor = transforms.ToTensor() #将数据转换成tensor形式
        self.transform_=transform_
        # 读取 csv 文件
        #利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  #header=None是去掉表头部分
        #read file name
        self.image_arr = np.asarray(self.data_info.iloc[1:, 0])  #self.data_info.iloc[1:,0表示读取第一列，从第二行开始一直读取到最后一行
        # read label
        self.label_arr = np.asarray(self.data_info.iloc[1:, 1])
        
 
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
 
    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]
 
        # 读取图像文件
        #img_as_img = Image.open(self.file_path + single_image_name + ".mrc")
        #img_as_img=mrcfile.open(self.file_path + single_image_name + ".mrc")
        img_as_img=mrcfile.open(os.path.join(self.file_path, single_image_name ))
        img_as_img=img_as_img.data
        
        #设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        #transform = transforms.Compose([
        #   transforms.Resize((224, 224)),
        #   transforms.ToTensor()
        #])
        transform=self.transform_
        img_as_img = transform(img_as_img)
 
        # 得到图像的 label
        label = self.label_arr[index]
        #print("index :{}   label: {}".format(index,label)) 
        return img_as_img, torch.tensor(int(label))  #返回每一个index对应的图片数据和对应的label
        
    def __len__(self):
        return self.data_len








def add_arguments(parser):
    parser.add_argument('-g', '--gpu', default=0, type=int, help='which device to use, set to -1 to force CPU (default: 0)')
    #parser.add_argument('-a', '--dir-a', nargs='+', help='directory of training images part A')
    #parser.add_argument('-b', '--dir-b', nargs='+', help='directory of training images part B')
    parser.add_argument('-d', '--data-dir',type=str, help='directory of training images')

    parser.add_argument('-i', '--data-name',type=str, help='path of csv file')
    parser.add_argument('-b', '--batch-size', type=int, help='batch size')
    
    parser.add_argument('-f', '--fewshot', type=int, help='number of training set')

    parser.add_argument('--save-prefix', type=str,help='path prefix to save denoising model')
    parser.add_argument('--num-classes', type=int,help='number of classes')
    parser.add_argument('--num-epochs',type=int, help='number of epoches')
    return parser





def draw_loss(train_loss,val_loss):
    fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    #pl.plot(x,y,'g-',label=u'train loss')`
    x1=range(len(train_loss))
    y1=train_loss

    x2=range(len(val_loss))
    y2=val_loss

    p2 = pl.plot(x1, y1,'r-', label = u'training')
    pl.legend()
    #显示图例
    p3 = pl.plot(x2,y2, 'b-', label = u'testing')
    pl.legend()
    pl.xlabel(u'epoches')
    pl.ylabel(u'loss')
    plt.title('Compare loss for training and validating')
    plt.savefig("train_results_loss_100.png")


def draw_acc(train_acc,val_acc):
    fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
    ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
    #pl.plot(x,y,'g-',label=u'train loss')`
    x1=range(len(train_acc))
    y1=train_acc

    x2=range(len(val_acc))
    y2=val_acc

    p2 = pl.plot(x1, y1,'r-', label = u'training')
    pl.legend()
    #显示图例
    p3 = pl.plot(x2,y2, 'b-', label = u'testing')
    pl.legend()
    pl.xlabel(u'epoches')
    pl.ylabel(u'loss')
    plt.title('Compare acc for training and validating')
    plt.savefig("train_results_acc_100.png")


def train_model(dataloaders,model, criterion, optimizer, scheduler, num_epochs=25,save_prefix=''):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    #for loss curve
    train_loss_list=[]
    val_loss_list = []
    train_acc_list=[]
    val_acc_list = []
    train_f1_list=[]
    val_f1_list = []



    best_epoch=0

    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

       
        running_loss = 0.0
        running_corrects = 0.0
        running_loss_val = 0.0
        running_corrects_val = 0.0
 

        f1_val=0.0
        best_f1_val=0.0
        f1_val=0.0
        best_f1_val=0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                #print("data:{}".format(data))  
                
                #print("label: {}".format(labels))
                # wrap them in Variable
                if use_cuda:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #f1_train+=f1_score(preds.cpu().numpy(),labels.cpu().numpy(), average='macro')
                    optimizer.step()
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                    #print("running_loss:  ",loss.item())
                if phase=='val':
                    #print(preds.cpu().numpy())
                    #print("val size: {}".format(len(preds.cpu().numpy())))
                    f1_val+=f1_score(preds.cpu().numpy(),labels.cpu().numpy(), average='macro')*len(preds.cpu().numpy())
                    running_loss_val += loss.item()
                    running_corrects_val += torch.sum(preds == labels.data).to(torch.float32)
                    #print("running_loss_val:  ",loss.item())
            scheduler.step()

        epoch_f1_val=f1_val/dataset_sizes['val']  #batch size is 20
        if epoch_f1_val>best_f1_val:
            best_f1_val=epoch_f1_val
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        epoch_loss_val = running_loss_val / dataset_sizes['val']
        epoch_acc_val = running_corrects_val / dataset_sizes['val']
        train_loss_list.append(epoch_loss)
        val_loss_list.append(epoch_loss_val)
        train_acc_list.append(epoch_acc)
        val_acc_list.append(epoch_acc_val)
        val_f1_list.append(epoch_f1_val)
        #print('train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        #print('val Loss: {:.4f} Acc: {:.4f}'.format( epoch_loss_val, epoch_acc_val))
            # deep copy the model
        if phase == 'val' and epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = model.state_dict()
            path = save_prefix + '_epoch{}.sav'.format(epoch)
            torch.save(model, path)
            best_epoch=epoch   
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}   Best epoch : {}'.format(best_acc,best_epoch))
    print('Best val F1: {:4f}'.format(best_f1_val))
    #print('loss len: {}  acc len:{}'.format(len(train_loss_list),len(train_acc_list)))
    
    #draw_loss(train_loss_list,val_loss_list)
    #draw_acc(train_acc_list,val_acc_list)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_acc,best_f1_val

def construct_random_trainingset(traingset_path,data_name,fewshot=20):
    #traingset_path=os.path.join(data_dir,'train_all.csv')
    data_info = pd.read_csv(traingset_path, header=None)
    image_arr = np.asarray(data_info.iloc[1:, 0])
    label_arr = np.asarray(data_info.iloc[1:, 1])
    all_data_len=len(image_arr)
    print("all_data:{}  fewshot:{}".format(all_data_len,fewshot))
    sample_index_1=random.sample(range(1,int(all_data_len/3)),fewshot)
    sample_index_2=random.sample(range(int(all_data_len/3),int(all_data_len/3)*2),fewshot)
    sample_index_3=random.sample(range(int(all_data_len/3)*2,all_data_len),fewshot)
    print("sample_index_1:  {}".format(sample_index_1))
    print("sample_index_2:  {}".format(sample_index_2))
    print("sample_index_3:  {}".format(sample_index_3))
    filename_train = os.path.join(data_dir,data_name+'_train.csv')
    filename_val = os.path.join(data_dir,data_name+'_val.csv')
    f_train = open(filename_train,'w')
    writer_train = csv.writer(f_train)
    f_val = open(filename_val,'w')
    writer_val = csv.writer(f_val)
    data_row=['filename','label']
    writer_train.writerow(data_row)
    writer_val.writerow(data_row)
    for i in range(all_data_len):
        if i in sample_index_1:
            data_row=[image_arr[i],label_arr[i]]
            writer_train.writerow(data_row)
        elif i in sample_index_2:
            #i=i+int(all_data_len/3)
            data_row=[image_arr[i],label_arr[i]]
            writer_train.writerow(data_row)
        elif i in sample_index_3:
            #i=i+int(all_data_len/3)*2
            data_row=[image_arr[i],label_arr[i]]
            writer_train.writerow(data_row)
        else:
            data_row=[image_arr[i],label_arr[i]]
            writer_val.writerow(data_row)
    f_train.close()
    f_val.close()



if __name__ == '__main__':

    parser=argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()

    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            #transforms.Resize(299),
            #transforms.CenterCrop(224),
           # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
           # transforms.Normalize([0.485], [0.229])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Scale(256),
            #transforms.Resize(299),  
            #transforms.CenterCrop(224),
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            #transforms.Normalize([0.485], [0.229])
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # your image data file

    data_dir=args.data_dir
    data_name=args.data_name

    #print('data_dir:{}'.format(data_dir))
 
    batch_size=args.batch_size
    fewshot=args.fewshot


    #dataset name of each dataset  example:10406_train.csv 10406_val.csv 10406_all.csv
    all_filename=os.path.join(data_dir,data_name+'_all.csv')
    train_filename=os.path.join(data_dir,data_name+'_train.csv')
    val_filename=os.path.join(data_dir,data_name+'_val.csv')
    #tmp_filename=os.path.join(data_dir,file_name+'_tmp.csv')
    #all_filename=os.path.join(data_dir,file_name+'_all.csv')
    #data_dir=os.path.join(data_dir,file_name)
    #construct_random_trainingset(all_filename,data_name,fewshot)
    
    '''
    TrainDataset = CreateDatasetFromImages(os.path.join(data_dir,'train.csv'),os.path.join(data_dir,'images'),data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainDataset,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    ValDataset = CreateDatasetFromImages(os.path.join(data_dir,'val.csv'),os.path.join(data_dir,'images'),data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(
        dataset=ValDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    dataloaders={'train':train_loader,'val':val_loader}
    '''


    #dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    '''
    dataset_size_train=TrainDataset.data_len
    dataset_size_val=ValDataset.data_len
    print('dataset_size_train: {}   dataset_size_val:{}'.format(dataset_size_train,dataset_size_val))
    dataset_sizes={'train':dataset_size_train,'val':dataset_size_val}
    '''
    '''
    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    num_classes=args.num_classes
    model_ft.fc = nn.Linear(num_ftrs, num_classes)


    device=args.gpu
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(device)
    #use_cuda = topaz.cuda.set_device(device)
    
    print('# using device={} with cuda={}'.format(device, use_cuda), file=sys.stderr)

    if use_cuda:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    '''
    

    #test for 1w iterations
    best_acc_all=0
    best_accs=[]
    best_f1s=[]
    iterations=2000
    for i in tqdm(range(iterations)):
        construct_random_trainingset(all_filename,data_name,fewshot)
        TrainDataset = CreateDatasetFromImages(train_filename,os.path.join(data_dir,'images'),data_transforms['train'])
        train_loader = torch.utils.data.DataLoader(
            dataset=TrainDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        ValDataset = CreateDatasetFromImages(val_filename,os.path.join(data_dir,'images'),data_transforms['val'])
        val_loader = torch.utils.data.DataLoader(
            dataset=ValDataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        dataloaders={'train':train_loader,'val':val_loader}
        
        dataset_size_train=TrainDataset.data_len
        dataset_size_val=ValDataset.data_len
        #print('dataset_size_train: {}   dataset_size_val:{}'.format(dataset_size_train,dataset_size_val))
        dataset_sizes={'train':dataset_size_train,'val':dataset_size_val}
        
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        num_classes=args.num_classes
        model_ft.fc = nn.Linear(num_ftrs, num_classes)



        #for name, param in model_ft.named_parameters():
        #    print(name,param)

        device=args.gpu
        use_cuda = torch.cuda.is_available()
        torch.cuda.set_device(device)
        #use_cuda = topaz.cuda.set_device(device)

        print('# using device={} with cuda={}'.format(device, use_cuda), file=sys.stderr)

        if use_cuda:
            model_ft = model_ft.cuda()

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        

        model_ft_,best_acc,best_f1 = train_model(dataloaders=dataloaders,model=model_ft,criterion=criterion,optimizer=optimizer_ft, scheduler=exp_lr_scheduler,num_epochs=args.num_epochs,save_prefix=args.save_prefix)
    
        best_accs.append(best_acc)
        best_f1s.append(best_f1)
        if best_acc >best_acc_all:
            torch.save(model_ft_, args.save_prefix + '_iter'+str(i)+'_best.sav')
        best_acc_ave_=sum(best_accs)/(i+1)
        best_f1_ave_=sum(best_f1s)/(i+1)
        print("{} iterations done! best_acc average is {}".format(i+1,best_acc_ave_))
        print("{} iterations done! best_f1 average is {}".format(i+1,best_f1_ave_))
        torch.cuda.empty_cache()
    best_acc_ave=sum(best_accs)/iterations
    best_f1_ave=sum(best_f1s)/iterations
    print("2000 iterations done! best_acc average is {}".format(best_acc_ave))
    print("2000 iterations done! best_f1 average is {}".format(best_f1_ave))
