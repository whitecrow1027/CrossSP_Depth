import os
import torch
import torchvision.transforms as transforms
from PIL import Image


def mkdir(path):
    if os.path.exists(path):
        #print("path %s existed!" %(path))
        pass
    else:
        os.makedirs(path)
        print("make path %s " %(path)) 

def tensor2img(img_tensor):
    image = img_tensor.clone().cpu()
    #image = image.squeeze()
    image = 0.5*(image.data+1)
    #print("image: type %s size %s" %(image.type(),image.size()))
    image = transforms.ToPILImage()(image)
    return image

class ImageInfo():
    """
    init: name[str],imgTensor[tensor, c*H*W]
    """
    def __init__(self,name,imgTensor):
        self.name = name
        self.imgTensor = imgTensor
        self.img = tensor2img(self.imgTensor)

    def show(self):
        pass

    def save(self,saveDir):
        imgName = saveDir+'/%s' %(self.name)
        self.img.save(imgName)

class ImageSaving():
    """
    A image buffer for image save
    path: root path that image saved
    ext: image format. eg: '.png'(default)
    """
    def __init__(self,path,ext='png'):
        self.savepath = path
        self.ext = ext
        self.image = []
        mkdir(self.savepath)

    def addImage(self,name,imgTensor):
        image = ImageInfo(name+self.ext,imgTensor)
        self.image.append(image)

    def imgClear(self):
        self.image = []
    
    def imgSave(self):
        for img in self.image:
            img.save(self.savepath)
        self.imgClear()

class updateLr():

    def __init__(self,num_epochs,decay_epoch):
        assert ((num_epochs - decay_epoch) > 0), "Decay must start before the training ends!"
        self.num_epochs=num_epochs
        self.decay_epoch=decay_epoch
    
    def update(self,epoch):
        return 1.0 - max(0,epoch-self.decay_epoch)/float(self.num_epochs-self.decay_epoch)

