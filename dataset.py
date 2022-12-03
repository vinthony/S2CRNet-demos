from __future__ import print_function, absolute_import


import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os.path import join
from PIL import Image


class HarmonizationDataset(data.Dataset):
    def __init__(self, datapath):
        self.train = []
        self.anno = []
        self.mask = []
        self.file_names = []
        self.labels = []
        
        input_size = 256

        self.file_list = join(datapath,'labels.txt')
            
        with open(self.file_list) as f:
            for line in f.readlines():
                self.file_names.append(line.rstrip().split(' ')[0])
                self.labels.append(int(line.rstrip().split(' ')[2]))
            

        for file_name in self.file_names: # prepare the pair of filenames
            self.train.append(join(datapath, 'composite_images', file_name))

            mask_path = join(datapath, 'masks', file_name).replace(('_'+file_name.split('_')[-1]),'.png')
            self.mask.append(mask_path)
            
            anno_path = join(datapath, 'real_images', file_name).replace(('_'+file_name.split('_')[-2]+'_'+file_name.split('_')[-1]),'.jpg')
            self.anno.append(anno_path)

        self.hr_trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])

        self.trans_tensor = transforms.Compose([
            transforms.ToTensor()])

        self.lr_trans = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor()
            ])

        print('total Dataset of is :', len(self.train))
        print('train:', len(self.train), 'mask:', len(self.mask), 'anno:', len(self.anno))

    def __getitem__(self, item):

        img = Image.open(self.train[item]).convert('RGB')
        mask = Image.open(self.mask[item]).convert('L')
        anno = Image.open(self.anno[item]).convert('RGB')
      
        bbox = mask.getbbox() # (left, upper, right, lower)

        thumb_foreground = img.crop(bbox)
        thumb_mask = mask.crop(bbox)


        return {"composite_images": self.lr_trans(img),
                "real_images": self.lr_trans(anno),
                "mask": self.lr_trans(mask),
                "fore_images": self.lr_trans(thumb_foreground),
                "fore_mask":self.lr_trans(thumb_mask),
                "label": self.labels[item],
                "name": self.train[item].split('/')[-1],
                "ori_img": self.hr_trans(img),
               }


    def __len__(self):
        return len(self.train)

