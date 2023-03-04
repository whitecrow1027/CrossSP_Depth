import torch
import argparse
import time
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

from datasets import CreatDatasets_msc
from model_mscdepth import ResnetEncoder,DepthDecoder
from reprojection import Reprojection
from evaluation import evaluate_with_gt
from utils import ImageSaving

if __name__ == "__main__":
    #print('pytorch ',torch.__version__)
    # Argument parsing
    parser = argparse.ArgumentParser(description='pytorch ddp training template')
    # device argument for torch.distributed.launch auto-assignment for multi-gpus training
    parser.add_argument('--name_exp', type=str,default=time.strftime('%Y_%m_%d_%H_%M'),help='name of the experiment to save')
    # parser.add_argument('--msclist',type=str,default='data/msc_stereo_new.txt',help='dataset list name')
    parser.add_argument('--msclist',type=str,default='data/new.txt',help='dataset list name')
    parser.add_argument('--mscroot',type=str,default='/opt/data/common/MSC',help='MSC root path')
    parser.add_argument('--kittilist',type=str,default='data/kitti.txt',help='dataset list name')
    parser.add_argument('--kittiroot',type=str,default='/opt/data/common/common/Kitti/Kitti_raw_data/',help='kitti root path')
    parser.add_argument('--imagesize',type=int ,default=[320,512],help='imagesize to resize')
    parser.add_argument('--res_num',type=int ,default=18,help='resnet type 18/50')
    parser.add_argument('--img_save_frq',type=int,default=1,help="image save frequency during val and test")

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--n_threads', type=int, default=16, help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=31415, help='Pseudo-RNG seed')

    # log args
    parser.add_argument('--logdir',type=str, default='./log',help='training log and mode saved dir')
    parser.add_argument('--test_epoch',type=int, default=30,help='model epoch num')
    parser.add_argument('--pretrain',type=str,default="log/2022_07_13_03_43",help='pretrain model dir')
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
        exit("No pretrain model")
    else:
        logdir = args.pretrain+'/test_show'
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logfile = logdir+'/log.txt'

    # models
    netDisp_encode_RGB = ResnetEncoder(num_layers=args.res_num,pretrained=True,num_input_images=1)
    netDisp_encode_IR = ResnetEncoder(num_layers=args.res_num,pretrained=True,num_input_images=1)
    netDisp_decode = DepthDecoder(netDisp_encode_RGB.num_ch_enc,scales=range(4),num_output_channels=1)

    # if load checkpoint
    checkpointpath=args.pretrain+'/checkpoint'
    netDisp_encode_RGB.load_state_dict(torch.load('%s/%s_netDisp_encode_RGB.pth'%(checkpointpath,args.test_epoch)))
    netDisp_encode_IR.load_state_dict(torch.load('%s/%s_netDisp_encode_IR.pth'%(checkpointpath,args.test_epoch)))
    netDisp_decode.load_state_dict(torch.load('%s/%s_netDisp_decode.pth'%(checkpointpath,args.test_epoch)))


    # send model to gpu
    netDisp_encode_RGB = netDisp_encode_RGB.to(args.device)
    netDisp_encode_IR = netDisp_encode_IR.to(args.device)
    netDisp_decode = netDisp_decode.to(args.device)

    
    val_dataset = CreatDatasets_msc(imgsize=args.imagesize,msc_list=args.msclist,rootpath=args.mscroot,mode='test')
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.n_threads,
                                shuffle=False)
    
    # create summary writer
    #log_writer = SummaryWriter(logdir)
    warp = Reprojection()
    # training loop
    with torch.no_grad():
        # validation
        netDisp_encode_RGB.eval()
        netDisp_encode_IR.eval()
        netDisp_decode.eval()

        datalist = enumerate(val_dataloader)
        save_dir = logdir
        datalist = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        img_saving_buffer = ImageSaving(save_dir+'/val','.png')
        logfile = save_dir+'/val_log.txt'
        logs=[]
        running_total_loss=0
        for id, mini_batch in datalist:
            IR_L = mini_batch['IR_L'].to(args.device)
            RGB_R = mini_batch['RGB_R'].to(args.device)
            
            depth_L = mini_batch['Depth_L'].to(args.device)
            depth_R = mini_batch['Depth_R'].to(args.device)

            F_msc_L = netDisp_encode_IR(IR_L)
            F_msc_R = netDisp_encode_RGB(RGB_R)
            disps_MSC_L = netDisp_decode(F_msc_L)
            disps_MSC_R = netDisp_decode(F_msc_R)
            

            IR_R_msc = warp(IR_L,disps_MSC_R[3],mode='right')
            RGB_L_msc = warp(RGB_R,disps_MSC_L[3],mode='left')

            disp_rgb_R = disps_MSC_R[3]
            rgb_abs_rel, rgb_sq_rel, rgb_rmse, rgb_rmse_log, rgb_a1, rgb_a2, rgb_a3 = evaluate_with_gt(depth_R,disp_rgb_R)
            
            disp_ir_L = disps_MSC_L[3]
            ir_abs_rel, ir_sq_rel, ir_rmse, ir_rmse_log, ir_a1, ir_a2, ir_a3 = evaluate_with_gt(depth_L,disp_ir_L)

            running_total_loss += ir_a1.item()
            logs_rgb = {'rgb_abs_rel':rgb_abs_rel,'rgb_sq_rel':rgb_sq_rel,'rgb_rmse':rgb_rmse,'rgb_rmse_log':rgb_rmse_log,'rgb_a1':rgb_a1,'rgb_a2':rgb_a2,'rgb_a3':rgb_a3}
            logs_ir = {'ir_abs_rel':ir_abs_rel,'ir_sq_rel':ir_sq_rel,'ir_rmse':ir_rmse,'ir_rmse_log':ir_rmse_log,'ir_a1':ir_a1,'ir_a2':ir_a2,'ir_a3':ir_a3}
            log = {**logs_rgb,**logs_ir}
            logs.append(log)
            with open(logfile,'a') as f:
                write_log = "id {}".format(id)+str(log)+'\n'
                f.write(write_log)

            datalist.set_description(
            ' validation acc mean/step: %.3f/%.3f' % (running_total_loss / (id + 1),
                                            ir_a1.item()))
            
            if id%args.img_save_frq==0:
                img_saving_buffer.addImage('%s_disp_msc_L'%(id),3.333*disp_ir_L[0])
                img_saving_buffer.addImage('%s_disp_msc_R'%(id),3.333*disp_rgb_R[0])
                img_saving_buffer.addImage('%s_RGB_R'%(id),RGB_R[0])
                img_saving_buffer.addImage('%s_IR_L'%(id),IR_L[0])
                img_saving_buffer.imgSave()
        
        # write log info and checkpoint
        rgb_abs_rel = np.stack([x['rgb_abs_rel'] for x in logs]).mean()
        rgb_sq_rel = np.stack([x['rgb_sq_rel'] for x in logs]).mean()
        rgb_rmse = np.stack([x['rgb_rmse'] for x in logs]).mean()
        rgb_rmse_log = np.stack([x['rgb_rmse_log'] for x in logs]).mean()
        rgb_a1 = np.stack([x['rgb_a1'] for x in logs]).mean()
        rgb_a2 = np.stack([x['rgb_a2'] for x in logs]).mean()
        rgb_a3 = np.stack([x['rgb_a3'] for x in logs]).mean()
        fin_logs_rgb = {'rgb_abs_rel':rgb_abs_rel,'rgb_sq_rel':rgb_sq_rel,'rgb_rmse':rgb_rmse,'rgb_rmse_log':rgb_rmse_log,'rgb_a1':rgb_a1,'rgb_a2':rgb_a2,'rgb_a3':rgb_a3}
        fin_logs_rgb = 'final avg rgb: ' +str(fin_logs_rgb) +'\n'

        ir_abs_rel = np.stack([x['ir_abs_rel'] for x in logs]).mean()
        ir_sq_rel = np.stack([x['ir_sq_rel'] for x in logs]).mean()
        ir_rmse = np.stack([x['ir_rmse'] for x in logs]).mean()
        ir_rmse_log = np.stack([x['ir_rmse_log'] for x in logs]).mean()
        ir_a1 = np.stack([x['ir_a1'] for x in logs]).mean()
        ir_a2 = np.stack([x['ir_a2'] for x in logs]).mean()
        ir_a3 = np.stack([x['ir_a3'] for x in logs]).mean()
        fin_logs_ir = {'ir_abs_rel':ir_abs_rel,'ir_sq_rel':ir_sq_rel,'ir_rmse':ir_rmse,'ir_rmse_log':ir_rmse_log,'ir_a1':ir_a1,'ir_a2':ir_a2,'rgb_a3':ir_a3}
        fin_logs_ir = 'final avg ir: ' +str(fin_logs_ir) +'\n'
        #print('test loss: ',avg_loss)
        with open(logfile,'a') as f:
            f.write(fin_logs_rgb)
            f.write(fin_logs_ir)