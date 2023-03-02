import torch
import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):

    def __init__(self,input_nc):
        super(Resblock,self).__init__()
        conv_net = [    nn.ReflectionPad2d(1),
                        nn.Conv2d(input_nc,input_nc,kernel_size=3),
                        nn.InstanceNorm2d(input_nc),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(input_nc,input_nc,kernel_size=3),
                        nn.InstanceNorm2d(input_nc)
                    ]
        self.conv_net = nn.Sequential(*conv_net)

    def forward(self,x):
        #return F.relu(self.conv_net(x) + x,inplace=True)
        return self.conv_net(x) + x


class Generator(nn.Module):

    def __init__(self,input_nc,output_nc,ngf,n_downsampling=2,n_resblock=9):
        super(Generator,self).__init__()

        #input convolution layer
        model = [   nn.Conv2d(input_nc, ngf, kernel_size=7,stride=1,padding=3),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(inplace=True)
                ]   

        #downsampling
        for i in range(n_downsampling):
            multi = 2**i 
            model += [  nn.Conv2d(ngf*multi,ngf*multi*2,kernel_size=3,padding=1,stride=2),
                        nn.InstanceNorm2d(ngf*multi*2),
                        nn.ReLU(inplace=True)
                    ]

        #residual blocks
        for i in range(n_resblock):
            model += [ Resblock(ngf*2**n_downsampling) ]
        
        #upsampling
        for i in range(n_downsampling):
            multi = 2**(n_downsampling-i)
            model += [  nn.ConvTranspose2d(ngf*multi,int(ngf*multi/2),kernel_size=3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(int(ngf*multi/2)),
                        nn.ReLU(inplace=True),
                    ]

        #output convolution layer
        model += [  nn.Conv2d(64,output_nc,kernel_size=7,stride=1,padding=3),
                    nn.Tanh()
                ]

        self.model = nn.Sequential(*model)
        
        self.apply(weights_init)

    def forward(self,x):
        return self.model(x)

class Discriminator(nn.Module):

    def __init__(self,input_nc,ndf,down_sample=3):
        super(Discriminator,self).__init__()
        
        #input layer
        model = [   nn.Conv2d(input_nc,ndf,kernel_size=4,stride=2,padding=1),
                    nn.LeakyReLU(0.2,inplace=True)
                ]

        #convolution layer
        for i in range(down_sample):
            multi=2**i
            model += [  nn.Conv2d(ndf*multi,ndf*multi*2,kernel_size=4,stride=2,padding=1),
                        nn.InstanceNorm2d(ndf*multi*2),
                        nn.LeakyReLU(0.2,inplace=True)
                     ]
        
        #output layer
        multi = 2**down_sample
        model += [  nn.Conv2d(ndf*multi,1,kernel_size=4,stride=1,padding=0)
                    #nn.Sigmoid()
                 ]

        self.model = nn.Sequential(*model)

        self.apply(weights_init)

    def forward(self,x):
        x = self.model(x)
        #return F.avg_pool2d(x,x.size()[2:]).view(x.size()[0],-1)
        return x.squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class updateLr():

    def __init__(self,num_epochs,decay_epoch):
        assert ((num_epochs - decay_epoch) > 0), "Decay must start before the training ends!"
        self.num_epochs=num_epochs
        self.decay_epoch=decay_epoch
    
    def update(self,epoch):
        return 1.0 - max(0,epoch-self.decay_epoch)/float(self.num_epochs-self.decay_epoch)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

class Generator_encoder(nn.Module):

    def __init__(self,input_nc,ngf,n_downsampling=2,n_resblock=4):
        super(Generator_encoder,self).__init__()

        #input convolution layer
        model = [   nn.Conv2d(input_nc, ngf, kernel_size=7,stride=1,padding=3),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(inplace=True)
                ]   

        #downsampling
        for i in range(n_downsampling):
            multi = 2**i 
            model += [  nn.Conv2d(ngf*multi,ngf*multi*2,kernel_size=3,padding=1,stride=2),
                        nn.InstanceNorm2d(ngf*multi*2),
                        nn.ReLU(inplace=True)
                    ]

        #residual blocks
        for i in range(n_resblock):
            model += [ Resblock(ngf*2**n_downsampling) ]
        
        self.model = nn.Sequential(*model)
        
        self.apply(weights_init)

    def forward(self,x):
        return self.model(x)

class Generator_decoder(nn.Module):

    def __init__(self,output_nc,ngf,n_downsampling=2,n_resblock=5):
        super(Generator_decoder,self).__init__()

        model = []
        #residual blocks
        for i in range(n_resblock):
            model += [ Resblock(ngf*2**n_downsampling) ]
        
        #upsampling
        for i in range(n_downsampling):
            multi = 2**(n_downsampling-i)
            model += [  nn.ConvTranspose2d(ngf*multi,int(ngf*multi/2),kernel_size=3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(int(ngf*multi/2)),
                        nn.ReLU(inplace=True),
                    ]

        #output convolution layer
        model += [  nn.Conv2d(64,output_nc,kernel_size=7,stride=1,padding=3),
                    nn.Tanh()
                ]
        
        self.model = nn.Sequential(*model)
        
        self.apply(weights_init)
    
    def forward(self, x):
        return self.model(x)