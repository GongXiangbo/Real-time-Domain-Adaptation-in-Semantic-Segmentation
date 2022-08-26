import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import torchvision.transforms
from PIL import Image
from ptflops import get_model_complexity_info

from models.build_BiSeNet import BiSeNet
from models.discriminator import LightDiscriminator
from models.utils import change_normalization_layer
from scripts.train import train, continue_train
from scripts.eval import test
from utils.config import Options
from utils.metrics import StreamSegMetrics
import utils.transform as t
from dataset.utils import find_dataset_using_name
from utils.iter_counter import IterationCounter
from utils.visualizer import Visualizer
from dataset.utils import encode_segmap


#Load the target source
image_t_dir = './data/cityscapes/images/'
file_t_list = './data/cityscapes/val.txt'
img_t_ids = [line.split('/')[0] + '_' + line.split('_')[1] + '_' + line.split('_')[2] for line in tuple(open(file_t_list, 'r'))]
image_t_list = [image_t_dir + ids + '_leftImg8bit.png' for ids in img_t_ids]


def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    return x_mean, x_std


class GTA5(Dataset):
    def __init__(self, root, mean, crop_size, transform=None, max_iters=None, ignore_index=255):
        super(GTA5, self).__init__()
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.ignore_index = ignore_index
        self.mean = mean
        self.files = []

        self.image_dir = os.path.join(self.root, 'images/') #image address
        self.label_dir = os.path.join(self.root, 'labels/') #label address
        #Get the txt file of the training images names
        file_list = os.path.join(self.root, 'train.txt')
        self.img_ids = [line.split('.')[0] for line in tuple(open(file_list, 'r'))]
        # import the class mapping
        self.info = json.load(open('./data/GTA5/info.json', 'r'))
        self.class_mapping = self.info['label2train']

        for name in self.img_ids:
          image_path = os.path.join(self.image_dir + name + '.png')
          label_path = os.path.join(self.label_dir + name + '.png')
          self.files.append({
              "image": image_path,
              "label": label_path,
              "name": name
          })


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']
        
        #LAB based image translation
        image_t = Image.open(random.choice(image_t_list)).convert('RGB')
        image_s = np.asarray(image)
        image_t = np.asarray(image_t)

        image_s_lab = cv2.cvtColor(image_s, cv2.COLOR_RGB2LAB)
        image_t_lab = cv2.cvtColor(image_t, cv2.COLOR_RGB2LAB)

        s_mean, s_std = get_mean_and_std(image_s_lab)
        t_mean, t_std = get_mean_and_std(image_t_lab)

        image_s_lab = (((image_s_lab - s_mean) * t_std) / s_std) + t_mean
        image_s_lab[image_s_lab < 0] = 0
        image_s_lab[image_s_lab > 255] = 255
        image_s_lab = image_s_lab.round().astype(np.uint8)

        # height, width, channel = image_s_lab.shape
        # for i in range(0, height):
        #   for j in range(0, width):
        #     for k in range(0, channel):
        #       x = image_s_lab[i, j, k]
        #       x = ((x - s_mean[k]) * (t_std[k] / s_std[k])) + t_mean[k]
        #       # round or +0.5
        #       x = round(x)
        #       # boundary check
        #       x = 0 if x<0 else x
        #       x = 255 if x>255 else x
        #       image_s_lab[i, j, k] = x
        
        image_s_new = cv2.cvtColor(image_s_lab, cv2.COLOR_LAB2RGB)

        image = Image.fromarray(image_s_new)
        
        #transform
        if self.transform is not None:
          image, label = self.transform(image, label)

        # resize
        crop = torchvision.transforms.CenterCrop(self.crop_size)
        image = crop(image)
        label = crop(label)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label
        label = encode_segmap(label, self.class_mapping, self.ignore_index)

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
      
      
class CityScapes(Dataset):
    def __init__(self, root, mean, crop_size, transform=None, train=True, max_iters=None, ignore_index=255, ssl=None):
        super(CityScapes, self).__init__()
        self.root = root
        self.mean = mean
        self.crop_size = crop_size
        self.transform = transform
        self.train = train
        self.ignore_index = ignore_index
        self.files = []
        self.set = 'train' if self.train else 'val'
        self.ssl = ssl

        self.image_dir = os.path.join(self.root, 'images/') #image address
        self.label_dir = os.path.join(self.root, 'labels/') #label address

        if self.train == True:
          file_list = os.path.join(self.root, 'train.txt')
        else:
          file_list = os.path.join(self.root, 'val.txt')

        self.img_ids = [line.split('/')[0] + '_' + line.split('_')[1] + '_' + line.split('_')[2] for line in tuple(open(file_list, 'r'))]
        # import the class mapping
        self.info = json.load(open('./data/cityscapes/info.json', 'r'))
        self.class_mapping = self.info['label2train']

        for name in self.img_ids:
          image_path = os.path.join(self.image_dir + name + '_leftImg8bit.png')
          label_path = os.path.join(self.label_dir + name + '_gtFine_labelIds.png')
          self.files.append({
              "image": image_path,
              "label": label_path,
              "name": name
          })
        

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        file = self.files[index]

        # open image and label file
        image = Image.open(file['image']).convert('RGB')
        label = Image.open(file['label'])
        name = file['name']

        #transform
        if self.transform is not None:
          image, label = self.transform(image, label)

        # resize
        crop = torchvision.transforms.CenterCrop(self.crop_size)
        if "train" in self.set:
            image = crop(image)
            label = crop(label)
        
        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # remap the semantic label
        if not self.ssl:
            label = encode_segmap(label, self.class_mapping, self.ignore_index)

        size = image.shape
        image = image[:, :, ::-1]
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name
      
      
## Use normal discriminator
def main():
    # Get arguments
    args = Options().parse()

    # Set cuda environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu_ids)

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize metric
    metrics = StreamSegMetrics(args, args.num_classes)

    # Initialize Visualizer
    visualizer = Visualizer(args)

    # Initialize Iteration counter
    iter_counter = IterationCounter(args)

    # Define/Load model
    model_d, model_dsp = None, None
    model = BiSeNet(num_classes=args.num_classes, context_path=args.context_path)

    # Define discriminators
    visualizer.info("Training with depthwise separable convolutional discriminator")
    model_d = LightDiscriminator(args.num_classes)

    model = torch.nn.DataParallel(model)
    model = model.to(args.gpu_ids[0])
    if model_d is not None:
        model_d = torch.nn.DataParallel(model_d)
        model_d = model_d.to(args.gpu_ids[0])
    if model_dsp is not None:
        torch.nn.DataParallel(model_dsp)
        model_dsp = model_dsp.to(args.gpu_ids[0])

    # Set cudnn
    cudnn.benchmark = True
    cudnn.enabled = True

    # Define data loaders
    transform = t.Compose([
        t.HorizontalFlip(),
        t.RandomScale((0.5, 0.75, 1, 1.5))
    ])
    gta5_train_dataset = GTA5(transform=transform, root=args.source_dataroot, mean=args.mean, crop_size=args.crop_size_source, max_iters=args.max_iters)
    cityscapes_val_dataset = CityScapes(root=args.target_dataroot, mean=args.mean, crop_size=args.crop_size_target, max_iters=args.max_iters, train=False, ssl=args.ssl)
    
    gta5_train_loader = DataLoader(gta5_train_dataset, batch_size=args.batch_size, shuffle=True)
    cityscapes_val_loader = DataLoader(cityscapes_val_dataset, batch_size=args.batch_size_val, shuffle=False)

    source_train_loader = gta5_train_loader
    target_train_loader = cityscapes_val_loader
    val_loader = cityscapes_val_loader
    test_loader = cityscapes_val_loader

    mean = args.mean_prep if args.use_st else args.mean
    if args.is_train:
        # Define source train loader
        source_train_loader = gta5_train_loader

        # Define val loader
        val_loader = cityscapes_val_loader
    else:
        # Define test loader
        test_loader = cityscapes_val_loader
    
    train(args, model, model_d, source_train_loader, target_train_loader, val_loader, metrics, iter_counter, visualizer)
    visualizer.close()
    
    
if __name__ == '__main__':
    main()

    
#mesure total Parameters and flops 
model_d = LightDiscriminator(19)
macs, params = get_model_complexity_info(model_d, (19, 1024, 512), as_strings=True, print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Flops: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
