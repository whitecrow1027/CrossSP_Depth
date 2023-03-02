import torch
from tqdm import tqdm
import torch.distributed as dist
import numpy as np

from loss import DispLoss,DispLoss_LR,Disp_consistency
# from reprojection import Reprojection
from reprojection import Reprojection
from evaluation import evaluate_with_gt
from utils import ImageSaving

def train_epoch(netDisp_encode_RGB,netDisp_encode_IR,netDisp_decode,netRec_RGB, netRec_IR ,netD,
                optimizer_disp_rgb,optimizer_disp_msc,optimizer_D, scheduler_D,scheduler_rgb,scheduler_ir,
                train_loader,args,epoch,logger,DDP=False):
    """
    training epoch for net
    """
    is_master = (not DDP) or args.is_master  #if not a ddp train or is master in ddp
    netDisp_encode_RGB.train()
    netDisp_encode_IR.train()
    netDisp_decode.train()
    netRec_RGB.train()
    netRec_IR.train()
    for d in netD:
        d.train()

    if DDP:
        dist.barrier()
    
    monodisp_loss = DispLoss(SSIM_w=args.w_ssim,disp_gradient_w=args.w_smooth,lr_w=args.w_lr_consis)
    LR_loss = DispLoss_LR(disp_gradient_w=args.lambda_disp_gradient/args.lambda_lr_consistency)
    rec_loss = torch.nn.L1Loss()
    gan_loss = torch.nn.MSELoss()
    disp_consistency_loss = Disp_consistency()
    warp = Reprojection()
    
    datalist = enumerate(train_loader)
    running_total_loss_disp = 0
    running_total_loss_msc = 0
    running_total_loss_D = 0
    
    if is_master:
        datalist = tqdm(enumerate(train_loader), total=len(train_loader))
        n_iter = epoch*len(train_loader)
    for id, mini_batch in datalist:
        ## load data
        IR_L_msc = mini_batch['IR_L'].to(args.device)
        RGB_R_msc = mini_batch['RGB_R'].to(args.device)
        VIS_L = mini_batch['VIS_L'].to(args.device)
        VIS_R = mini_batch['VIS_R'].to(args.device)

        ## rgb depth update
        optimizer_disp_rgb.zero_grad()
        F_VIS_L = netDisp_encode_RGB(VIS_L)
        F_VIS_R = netDisp_encode_RGB(VIS_R)
        disps_L = netDisp_decode(F_VIS_L)
        disps_R = netDisp_decode(F_VIS_R)
        # disps = []
        # for i in range(4):
        #     disps.append(torch.cat((disps_L[i],disps_R[i]),dim=1))
        loss_disp = monodisp_loss(disps_L,disps_R,[VIS_L,VIS_R])
        
        #print('loss_disp: ',loss_disp)
        loss_disp.backward()
        optimizer_disp_rgb.step()

        ## msc depth update
        optimizer_disp_msc.zero_grad()
        F_ir_L = netDisp_encode_IR(IR_L_msc)
        
        loss_F_ir = 0
        for i in range(5):
            IR_out = netD[i](F_ir_L[i])
            real_label = torch.tensor(1.0)
            real_label = real_label.cuda(RGB_R_msc.device.index)
            real_label=real_label.expand_as(IR_out)
            loss_F_ir = loss_F_ir+gan_loss(IR_out,real_label)*0.2

        F_rgb_R = netDisp_encode_RGB(RGB_R_msc)
        F_rgb_R = [F.detach() for F in F_rgb_R]   
        disps_L = netDisp_decode(F_ir_L)
        disps_R = netDisp_decode(F_rgb_R)

        # cross spectral reconstruction
        rec_RGB_R = netRec_RGB(F_rgb_R)
        rec_IR_L = netRec_IR(F_ir_L)
        loss_rec = rec_loss(rec_RGB_R,RGB_R_msc) + rec_loss(rec_IR_L,IR_L_msc)

        IR_R_msc = warp(IR_L_msc,disps_R[3],mode='right')
        RGB_L_msc = warp(RGB_R_msc,disps_L[3],mode='left')

        # L-R cycle and edge smooth
        loss_lr = LR_loss(disps_L,disps_R,IR_L_msc) 

        # spectral depth consistency

        F_rgb_L = netDisp_encode_RGB(RGB_L_msc)
        F_rgb_L = [F.detach() for F in F_rgb_L] 
        F_ir_R = netDisp_encode_IR(IR_R_msc)
        disps_L_rec = netDisp_decode(F_rgb_L)
        disps_R_rec = netDisp_decode(F_ir_R)
        loss_warp_consis = disp_consistency_loss(disps_L,disps_L_rec)+disp_consistency_loss(disps_R,disps_R_rec)

        loss_msc = loss_F_ir  + 0.1*loss_rec \
                + args.lambda_lr_consistency*loss_lr \
                + args.lambda_warp_consistency*loss_warp_consis
        
        loss_msc.backward()
        optimizer_disp_msc.step()

        ## adversial update
        optimizer_D.zero_grad()
        F_rgb = [F.detach() for F in F_VIS_L]    #save as real image
        F_ir = [F.detach() for F in F_ir_L]        #save as fake image
        loss_D = 0
        for i in range(5):
            real_label = torch.tensor(1.0)
            fake_label = torch.tensor(0.0)
            real_label = real_label.cuda(RGB_R_msc.device.index)
            fake_label = fake_label.cuda(RGB_R_msc.device.index)

            RGB_out = netD[i](F_rgb[i])
            real_label = real_label.expand_as(RGB_out)
            loss_D_real = gan_loss(RGB_out,real_label)
            IR_out = netD[i](F_ir[i])
            fake_label = fake_label.expand_as(IR_out)
            loss_D_fake = gan_loss(IR_out,fake_label)
            loss_D = loss_D + (loss_D_real+loss_D_fake)*0.5

        loss_D.backward()
        optimizer_D.step()

        running_total_loss_disp += loss_disp.item()
        running_total_loss_msc += loss_msc.item()
        running_total_loss_D += loss_D.item()
        if is_master: ### train log should only run on master
            logger.add_scalar('loss_msc',loss_msc.item(),n_iter)
            logger.add_scalar('loss_disp',loss_disp.item(),n_iter)
            logger.add_scalar('loss_D',loss_D.item(),n_iter)
            logger.add_scalar('loss_F_ir',loss_F_ir.item(),n_iter)
            logger.add_scalar('loss_rec',loss_rec.item(),n_iter)
            logger.add_scalar('loss_lr',loss_lr.item(),n_iter)
            logger.add_scalar('loss_warp_consis',loss_warp_consis.item(),n_iter)
            n_iter+=1
            datalist.set_description(
                'training: loss: disp/msc: %.3f/%.3f' % (running_total_loss_msc / (id + 1),
                                             running_total_loss_msc/ (id + 1)))
    running_total_loss_disp /= len(train_loader)
    running_total_loss_msc /= len(train_loader)
    running_total_loss_D /= len(train_loader)

    # update scheduler
    scheduler_D.step()
    scheduler_ir.step()
    scheduler_rgb.step()
    return running_total_loss_disp,running_total_loss_msc ,running_total_loss_D


def val_epoch(netDisp_encode_RGB,netDisp_encode_IR,netDisp_decode,netRec_RGB, netRec_IR ,netD,val_loader,args,epoch,save_dir,DDP=False):
    """
    valuation epoch for net
    """
    is_master = (not DDP) or args.is_master  #if not a ddp train or is master in ddp
    netDisp_encode_RGB.eval()
    netDisp_encode_IR.eval()
    netDisp_decode.eval()
    netRec_RGB.eval()
    netRec_IR.eval()
    for d in netD:
        d.eval()
    if DDP:
        dist.barrier()
    warp = Reprojection()
    
    running_total_loss = 0
    with torch.no_grad():
        datalist = enumerate(val_loader)
        if is_master:
            datalist = tqdm(enumerate(val_loader), total=len(val_loader))
            img_saving_buffer = ImageSaving(save_dir+'/val','.png')
            logfile = save_dir+'/val_log.txt'
            logs=[]
        for id, mini_batch in datalist:
            IR_L = mini_batch['IR_L'].to(args.device)
            RGB_R = mini_batch['RGB_R'].to(args.device)
            
            depth_L = mini_batch['Depth_L'].to(args.device)
            depth_R = mini_batch['Depth_R'].to(args.device)

            F_msc_L = netDisp_encode_IR(IR_L)
            F_msc_R = netDisp_encode_RGB(RGB_R)
            disps_MSC_L = netDisp_decode(F_msc_L)
            disps_MSC_R = netDisp_decode(F_msc_R)

            IR_R_warp = warp(IR_L,disps_MSC_R[3],mode='right')
            RGB_L_warp = warp(RGB_R,disps_MSC_L[3],mode='left')
            rec_RGB_L = netRec_RGB(F_msc_L)
            rec_IR_R= netRec_IR(F_msc_R)

            disp_rgb_R = disps_MSC_R[3]
            rgb_abs_rel, rgb_sq_rel, rgb_rmse, rgb_rmse_log, rgb_a1, rgb_a2, rgb_a3 = evaluate_with_gt(depth_R,disp_rgb_R)
            
            disp_ir_L = disps_MSC_L[3]
            ir_abs_rel, ir_sq_rel, ir_rmse, ir_rmse_log, ir_a1, ir_a2, ir_a3 = evaluate_with_gt(depth_L,disp_ir_L)

            running_total_loss += ir_a1.item()
            if is_master: ### val log should only run on master
                logs_rgb = {'rgb_abs_rel':rgb_abs_rel,'rgb_sq_rel':rgb_sq_rel,'rgb_rmse':rgb_rmse,'rgb_rmse_log':rgb_rmse_log,'rgb_a1':rgb_a1,'rgb_a2':rgb_a2,'rgb_a3':rgb_a3}
                logs_ir = {'ir_abs_rel':ir_abs_rel,'ir_sq_rel':ir_sq_rel,'ir_rmse':ir_rmse,'ir_rmse_log':ir_rmse_log,'ir_a1':ir_a1,'ir_a2':ir_a2,'ir_a3':ir_a3}
                log = {**logs_rgb,**logs_ir}
                logs.append(log)

                datalist.set_description(
                ' validation acc mean/step: %.3f/%.3f' % (running_total_loss / (id + 1),
                                             ir_a1.item()))
                
                if id%args.img_save_frq==0:
                    img_saving_buffer.addImage('%s_%s_disp_msc_L_test'%(epoch,id),3.333*disp_ir_L[0])
                    img_saving_buffer.addImage('%s_%s_disp_msc_R_test'%(epoch,id),3.333*disp_rgb_R[0])
                    img_saving_buffer.addImage('%s_%s_RGB_R_test'%(epoch,id),RGB_R[0])
                    img_saving_buffer.addImage('%s_%s_IR_L_test'%(epoch,id),IR_L[0])

                    img_saving_buffer.addImage('%s_%s_IR_R_warp_test'%(epoch,id),IR_R_warp[0])
                    img_saving_buffer.addImage('%s_%s_RGB_L_warp'%(epoch,id),RGB_L_warp[0])
                    img_saving_buffer.addImage('%s_%s_IR_R_rec_test'%(epoch,id),rec_IR_R[0])
                    img_saving_buffer.addImage('%s_%s_RGB_L_rec_test'%(epoch,id),rec_RGB_L[0])
                    img_saving_buffer.imgSave()
        
        ##epoch finished        
        if is_master:
            rgb_abs_rel = np.stack([x['rgb_abs_rel'] for x in logs]).mean()
            rgb_sq_rel = np.stack([x['rgb_sq_rel'] for x in logs]).mean()
            rgb_rmse = np.stack([x['rgb_rmse'] for x in logs]).mean()
            rgb_rmse_log = np.stack([x['rgb_rmse_log'] for x in logs]).mean()
            rgb_a1 = np.stack([x['rgb_a1'] for x in logs]).mean()
            rgb_a2 = np.stack([x['rgb_a2'] for x in logs]).mean()
            rgb_a3 = np.stack([x['rgb_a3'] for x in logs]).mean()
            fin_logs_rgb = {'rgb_abs_rel':rgb_abs_rel,'rgb_sq_rel':rgb_sq_rel,'rgb_rmse':rgb_rmse,'rgb_rmse_log':rgb_rmse_log,'rgb_a1':rgb_a1,'rgb_a2':rgb_a2,'rgb_a3':rgb_a3}
            fin_logs_rgb = 'epoch {} avg rgb: '.format(epoch) +str(fin_logs_rgb) +'\n'

            ir_abs_rel = np.stack([x['ir_abs_rel'] for x in logs]).mean()
            ir_sq_rel = np.stack([x['ir_sq_rel'] for x in logs]).mean()
            ir_rmse = np.stack([x['ir_rmse'] for x in logs]).mean()
            ir_rmse_log = np.stack([x['ir_rmse_log'] for x in logs]).mean()
            ir_a1 = np.stack([x['ir_a1'] for x in logs]).mean()
            ir_a2 = np.stack([x['ir_a2'] for x in logs]).mean()
            ir_a3 = np.stack([x['ir_a3'] for x in logs]).mean()
            fin_logs_ir = {'ir_abs_rel':ir_abs_rel,'ir_sq_rel':ir_sq_rel,'ir_rmse':ir_rmse,'ir_rmse_log':ir_rmse_log,'ir_a1':ir_a1,'ir_a2':ir_a2,'rgb_a3':ir_a3}
            fin_logs_ir = 'epoch {} avg ir: '.format(epoch) +str(fin_logs_ir) +'\n'
            #print('test loss: ',avg_loss)
            with open(logfile,'a') as f:
                f.write(fin_logs_rgb)
                f.write(fin_logs_ir)

        return running_total_loss / len(val_loader)
            