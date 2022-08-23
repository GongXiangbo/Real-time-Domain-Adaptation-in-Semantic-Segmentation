import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from loss import DiceLoss
import torch.cuda.amp as amp
from train import val, train
from PIL import Image
import torchvision.transforms
import cv2
import json
import transform as t


class CityScapesDataset(Dataset):
    def __init__(self, root, train=True, num_classes=19, transform=None):
      super(CityScapesDataset, self).__init__()
      self.root = root #存放数据集的地址
      self.train = train #是否为训练集
      self.num_classes = num_classes #数据集的类别数量
      self.transform = transform
      self._set_files()

      with open('./data/cityscapes/cityscapes_info.json', 'r') as fr:
        labels_info = json.load(fr)

      self.lb_map = {el['id']: el['trainId'] for el in labels_info}

    def _set_files(self): #获取数据集图片
      self.image_dir = os.path.join(self.root, 'images/') #图像地址
      self.label_dir = os.path.join(self.root, 'labels/') #label地址
      #获取训练或者验证的图像名字的txt文件
      if self.train == True:
        file_list = os.path.join(self.root, 'train.txt')
      else:
        file_list = os.path.join(self.root, 'val.txt')
      #获取对应图片的名字
      self.files = [line.split('/')[0] + '_' + line.split('_')[1] + '_' + line.split('_')[2] for line in tuple(open(file_list, 'r'))]

    def _load_data(self, index):
      image_id = self.files[index]
      image_path = os.path.join(self.image_dir + image_id + '_leftImg8bit.png')
      label_path = os.path.join(self.label_dir + image_id + '_gtFine_labelIds.png')
      image = Image.open(image_path).convert('RGB')
      label = Image.open(label_path)
      return image, label

    def __getitem__(self, index):
      image, label = self._load_data(index)
      im_lb = dict(im=image, lb=label)
      if self.transform is not None and self.train:
        im_lb = self.transform(im_lb)
      image, label = im_lb['im'], im_lb['lb']
      if self.train:
        crop_transform = torchvision.transforms.CenterCrop((512, 1024))
        image, label = crop_transform(image), crop_transform(label)
      to_tensor = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            ])
      image = to_tensor(image)
      label = np.array(label).astype(np.int64)[np.newaxis, :]
      label = self.convert_labels(label)
      # return image.detach(), label.unsqueeze(0).detach()
      return image, label

    def __len__(self):
      return len(self.files)

    def convert_labels(self, label):
      for k, v in self.lb_map.items():
          label[label == k] = v
      return label


def main(params, model):
    #load data
    transform_train = t.Compose([
      t.HorizontalFlip(),
      t.RandomScale((0.5, 0.75, 1.0, 1.5))
      # t.RandomCrop((1024, 1024))         
    ])
    train_dataset = CityScapesDataset(root='./data/cityscapes', train=True, transform=transform_train)
    val_dataset = CityScapesDataset(root='./data/cityscapes', train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # Create HERE datasets instance
    dataset_train = train_dataset
    dataset_val = val_dataset

    # Define HERE your dataloaders:
    dataloader_train = train_loader
    dataloader_val = val_loader

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # model = BiSeNet(args.num_classes, args.context_path)
    model = model
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)

    # val
    val(args, model, dataloader_val)
    

#use resnet101 as backone
params = [
    '--num_epochs', '50',
    '--learning_rate', '5e-4',
    # '--learning_rate', '0.01',
    '--data', './data/...',
    '--num_workers', '8',
    '--num_classes', '19',
    '--cuda', '0',
    '--batch_size', '4',
    '--save_model_path', './checkpoints_101_sgd',
    '--context_path', 'resnet101',  # set backone
    '--optimizer', 'sgd'

]
model = BiSeNet(19, 'resnet101')
main(params,model)


#use resnet18 as backone
params = [
    '--num_epochs', '50',
    '--learning_rate', '5e-4',
    # '--learning_rate', '0.01',
    '--data', './data/...',
    '--num_workers', '8',
    '--num_classes', '19',
    '--cuda', '0',
    '--batch_size', '4',
    '--save_model_path', './checkpoints_18_sgd',
    '--context_path', 'resnet18',  # set backone
    '--optimizer', 'sgd'

]
model = BiSeNet(19, 'resnet18')
main(params, model)
