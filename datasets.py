import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from evaluation_utils import load_gt_depth


class CreatDatasets_msc(Dataset):
    """
        creat a msc depth datasets by a list txt file
    """

    def __init__(self,imgsize,msc_list,transform=None,rootpath='/opt/data/common/common/MSC',mode='train'):
        super(CreatDatasets_msc,self).__init__()
        assert (mode == 'train' or mode == 'test') ,"error dataset mode"

        # crop  kitti image from 1242*375 to 600*375 and resize
        self.listfile_msc=msc_list.replace('.txt','_%s.txt'%mode)
        self.transform = transforms.Compose([transforms.Resize(imgsize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.list_RGB_R = []
        self.list_RGB_L = []
        self.list_IR_R = []
        self.list_IR_L =[]
        self.list_Depth_R =[]
        self.list_Depth_L =[]
        self.mode = mode
        self.rootpath=rootpath
        
        with open(self.listfile_msc) as f:
            list = f.readlines()
            for line in list:
                self.list_RGB_R.append(self.rootpath+line.split()[0])
                self.list_RGB_L.append(self.rootpath+line.split()[1])
                self.list_IR_R.append(self.rootpath+line.split()[2])
                self.list_IR_L.append(self.rootpath+line.split()[3])
                if self.mode == 'test':
                    self.list_Depth_R.append(self.rootpath+line.split()[4])
                    self.list_Depth_L.append(self.rootpath+line.split()[5])
    
    def __getitem__(self,index):
        if self.mode == 'train':    
            index_random = random.randint(0,len(self.list_RGB_R)-1)
            VIS_L = self.transform( Image.open(self.list_RGB_L[index_random]) )
            VIS_R = self.transform( Image.open(self.list_RGB_R[index_random]) )
            IR_L = self.transform( Image.open(self.list_IR_L[index]) )
            RGB_R = self.transform( Image.open(self.list_RGB_R[index]) )
            data={'VIS_L':VIS_L,'VIS_R':VIS_R,'IR_L':IR_L,'RGB_R':RGB_R}
        if self.mode == 'test':
            IR_L = self.transform( Image.open(self.list_IR_L[index]) )
            #IR_R = self.transform( Image.open(self.list_IR_R[index]) )
            RGB_R = self.transform( Image.open(self.list_RGB_R[index]) )
            RGB_L = self.transform( Image.open(self.list_RGB_L[index]) )
            Depth_L=load_gt_depth(self.list_Depth_L[index])
            Depth_R=load_gt_depth(self.list_Depth_R[index])
            data={'IR_L':IR_L,'RGB_R':RGB_R,'RGB_L':RGB_L,'Depth_L':Depth_L,'Depth_R':Depth_R}

        return data

    def __len__(self):
        return len(self.list_RGB_R)
