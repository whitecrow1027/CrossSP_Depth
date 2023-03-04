import torch
import argparse
import time
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets import CreatDatasets_msc
from model_mscdepth import ResnetEncoder,DepthDecoder,RecDecoder
from model_gan import Discriminator
from optimize import train_epoch,val_epoch

if __name__ == "__main__":
    #print('pytorch ',torch.__version__)
    # Argument parsing
    parser = argparse.ArgumentParser(description='pytorch training')
    # device argument for torch.distributed.launch auto-assignment for multi-gpus training
    parser.add_argument('--name_exp', type=str,default=time.strftime('%Y_%m_%d_%H_%M'),help='name of the experiment to save')
    parser.add_argument('--msclist',type=str,default='data/msc_stereo_new.txt',help='dataset list name')
    parser.add_argument('--mscroot',type=str,default='/opt/data/common/MSC',help='MSC root path')
    parser.add_argument('--imagesize',type=int ,default=[320,512],help='imagesize to resize')
    parser.add_argument('--res_num',type=int ,default=18,help='resnet type 18/50')
    parser.add_argument('--img_save_frq',type=int,default=300,help="image save frequency during val and test")

    # hparameters
    parser.add_argument('--lr',type=float ,default=0.0002,help='Learning rate for optimizers')
    parser.add_argument('--beta1',type=float ,default=0.5,help='Beta1 hyperparam for Adam optimizers')
    parser.add_argument('--lambda_disp_gradient',type=float,default=0.1,help='lambda for disp edge smooth')
    parser.add_argument('--lambda_lr_consistency',type=float,default=10,help='lambda for lr cycle loss')
    parser.add_argument('--lambda_warp_consistency',type=float,default=1,help='lambda for cycle depth consistency')
    parser.add_argument('--w_ssim',type=float,default=0.85,help="weight of ssim in disp loss")
    parser.add_argument('--w_smooth',type=float,default=0.5,help="weight of disp smooth in disp loss")
    parser.add_argument('--w_lr_consis',type=float,default=1,help="weight of lr consistency in disp loss")
    parser.add_argument('--ssim_kernel',type=float,default=3,help="kernel size of ssim opreator")

    # training parameters
    parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=60, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--scheduler_step', type=int, default=40, help='training batch size')
    parser.add_argument('--n_threads', type=int, default=16, help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=31415, help='Pseudo-RNG seed')

    # log args
    parser.add_argument('--logdir',type=str, default='./log',help='training log and mode saved dir')
    parser.add_argument('--pretrain',type=str,default=None,help='pretrain model dir')
    parser.add_argument('--info',type=str,default=None) 
    args = parser.parse_args()

    # get current device and weather master device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)

    # initialize random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logidr
    if not args.pretrain:
        logdir = args.logdir+'/'+args.name_exp
    else:
        logdir = args.pretrain
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logfile = logdir+'/log.txt'

    param_file=logdir+'/param.txt'
    param = str(args)
    with open(param_file,'a') as f:
        f.writelines(param+'\n')

    # models
    netDisp_encode_RGB = ResnetEncoder(num_layers=args.res_num,pretrained=True,num_input_images=1)
    netDisp_encode_IR = ResnetEncoder(num_layers=args.res_num,pretrained=True,num_input_images=1)
    netRec_RGB = RecDecoder(netDisp_encode_RGB.num_ch_enc,3,use_skips=True)
    netRec_IR = RecDecoder(netDisp_encode_IR.num_ch_enc,3,use_skips=True)
    netDisp_decode = DepthDecoder(netDisp_encode_RGB.num_ch_enc,scales=range(4),num_output_channels=1)
    netD1 = Discriminator(64,64,down_sample=3)
    netD2 = Discriminator(64,64,down_sample=3)
    netD3 = Discriminator(128,64,down_sample=2) 
    netD4 = Discriminator(256,128,down_sample=1)
    netD5 = Discriminator(512,256,down_sample=0)

    # if load checkpoint
    if args.pretrain:
        checkpointpath=logdir+'/checkpoint'
        netDisp_encode_RGB.load_state_dict(torch.load('%s/%s_netDisp_encode_RGB.pth'%(checkpointpath,args.start_epoch)))
        netDisp_encode_IR.load_state_dict(torch.load('%s/%s_netDisp_encode_IR.pth'%(checkpointpath,args.start_epoch)))
        netDisp_decode.load_state_dict(torch.load('%s/%s_netDisp_decode.pth'%(checkpointpath,args.start_epoch)))
        netRec_RGB.load_state_dict(torch.load('%s/%s_netRec_RGB.pth'%(checkpointpath,args.start_epoch)))
        netRec_IR.load_state_dict(torch.load('%s/%s_netRec_IR.pth'%(checkpointpath,args.start_epoch)))
        netD1.load_state_dict(torch.load('%s/%s_netD1.pth'%(checkpointpath,args.start_epoch)))
        netD2.load_state_dict(torch.load('%s/%s_netD2.pth'%(checkpointpath,args.start_epoch)))
        netD3.load_state_dict(torch.load('%s/%s_netD3.pth'%(checkpointpath,args.start_epoch)))
        netD4.load_state_dict(torch.load('%s/%s_netD4.pth'%(checkpointpath,args.start_epoch)))
        netD5.load_state_dict(torch.load('%s/%s_netD5.pth'%(checkpointpath,args.start_epoch)))
        print("training start from epoch %s" %(args.start_epoch))
    else:
        print("training start from begining")

    # send model to gpu
    netDisp_encode_RGB = netDisp_encode_RGB.to(args.device)
    netDisp_encode_IR = netDisp_encode_IR.to(args.device)
    netRec_RGB = netRec_RGB.to(args.device)
    netRec_IR = netRec_IR.to(args.device)
    netDisp_decode = netDisp_decode.to(args.device)
    netD1 = netD1.to(args.device)
    netD2 = netD2.to(args.device)
    netD3 = netD3.to(args.device)
    netD4 = netD4.to(args.device)
    netD5 = netD5.to(args.device)

    netD = [netD1,netD2,netD3,netD4,netD5]

    # train and val dataset
    train_dataset =  CreatDatasets_msc(imgsize=args.imagesize,msc_list=args.msclist,rootpath=args.mscroot,mode='train')
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.n_threads,
                                shuffle=True)
    
    val_dataset = CreatDatasets_msc(imgsize=args.imagesize,msc_list=args.msclist,rootpath=args.mscroot,mode='test')
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.n_threads,
                                shuffle=False)
    
    # Optimizer
    netD_parameters = [{'params':netD1.parameters(),
                            'params':netD2.parameters(),
                            'params':netD3.parameters(),
                            'params':netD4.parameters(),
                            'params':netD5.parameters()
                            }]
    optimizer_D = optim.Adam(netD_parameters,lr=args.lr,betas=(args.beta1,0.999))
    
    optimizer_disp_rgb = optim.Adam([{'params':netDisp_encode_RGB.parameters()},
                                {'params':netDisp_decode.parameters()}],
                                lr=args.lr,betas=(args.beta1,0.999))
    optimizer_disp_msc = optim.Adam([{'params':netRec_RGB.parameters()},
                                 {'params':netRec_IR.parameters()},
                                 {'params':netDisp_encode_IR.parameters()}],lr=args.lr,betas=(args.beta1,0.999))
    # create summary writer
    log_writer = SummaryWriter(logdir)

    ## learning rate scheduler
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, args.scheduler_step, 0.1)
    scheduler_rgb = optim.lr_scheduler.StepLR(optimizer_disp_rgb, args.scheduler_step, 0.1)
    scheduler_ir = optim.lr_scheduler.StepLR(optimizer_disp_msc, args.scheduler_step, 0.1)
    
    # training loop
    for epoch in range(args.start_epoch+1, args.n_epoch):
        print('starting epoch {}:'.format(epoch))
        # Training
        train_loss_disp,train_loss_msc,train_loss_D=train_epoch(netDisp_encode_RGB,netDisp_encode_IR,netDisp_decode,netRec_RGB, netRec_IR ,netD,
                                optimizer_disp_rgb,optimizer_disp_msc,optimizer_D,scheduler_D,scheduler_rgb,scheduler_ir,
                                train_dataloader,args,epoch,log_writer,DDP=False)
        
        # validation
        val_acc=val_epoch(netDisp_encode_RGB,netDisp_encode_IR,netDisp_decode,netRec_RGB, netRec_IR,netD,
                            val_dataloader,args,epoch,logdir,DDP=False)
        print("epoch {} val acc: {}".format(epoch,val_acc))           
        
        # write log info and checkpoint
        log_writer.add_scalar('loss_disp', train_loss_disp, epoch)
        log_writer.add_scalar('loss_msc', train_loss_msc, epoch)
        log_writer.add_scalar('loss_D', train_loss_D, epoch)
        log_writer.add_scalar('val_acc', val_acc, epoch)
        log = "epoch {} loss_disp {} loss_msc {} loss_D {}\n".format(epoch,train_loss_disp,train_loss_msc,train_loss_D)
        with open(logfile,'a') as f:
            f.write(log)
        if epoch > args.start_epoch:
            checkpointpath=logdir+'/checkpoint'
            if not os.path.isdir(checkpointpath):
                os.makedirs(checkpointpath)
            torch.save(netDisp_encode_RGB.state_dict(),'%s/%s_netDisp_encode_RGB.pth' % (checkpointpath,epoch))
            torch.save(netDisp_encode_IR.state_dict(),'%s/%s_netDisp_encode_IR.pth' % (checkpointpath,epoch))
            torch.save(netDisp_decode.state_dict(),'%s/%s_netDisp_decode.pth' % (checkpointpath,epoch))
            torch.save(netRec_RGB.state_dict(),'%s/%s_netRec_RGB.pth' % (checkpointpath,epoch))
            torch.save(netRec_IR.state_dict(),'%s/%s_netRec_IR.pth' % (checkpointpath,epoch))
            torch.save(netD1.state_dict(),'%s/%s_netD1.pth' % (checkpointpath,epoch))
            torch.save(netD2.state_dict(),'%s/%s_netD2.pth' % (checkpointpath,epoch))
            torch.save(netD3.state_dict(),'%s/%s_netD3.pth' % (checkpointpath,epoch))
            torch.save(netD4.state_dict(),'%s/%s_netD4.pth' % (checkpointpath,epoch))
            torch.save(netD5.state_dict(),'%s/%s_netD5.pth' % (checkpointpath,epoch))
