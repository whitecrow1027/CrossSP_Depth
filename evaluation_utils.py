import numpy as np
import torch
import torch.nn.functional as F
import cv2

def load_gt_depth(imgpath):
    """
    load origin 16-bit ground truth depth image
    """
    
    depth = cv2.imread(imgpath,-1)
    gt = depth.astype(np.float32)/256.0
    return gt

def depth2disp(depth,f=1634.5,B=0.5):
    """
    transfer depth image to disparity for pytorch tensor. Normalize to 0-1 by image width
    input depth image: 16bit image
    output disparity: 
    """
    depth=depth/256.0
    disp=f*B/depth
    width=disp.size(-1)
    disp = disp/width
    return disp

def disp2depth(disp,width,f=1634.5,B=0.5,max_depth=100,min_disp=0.01):
    """
    transfer disparity image to depth for pytorch tensor or numpy array.
    """
    disp=disp*width+min_disp
    depth = f*B/disp
    depth[depth>max_depth]=max_depth
    #depth = depth*256
    return depth

def get_depth_array(gt_depth,pred_disp):
    """
    convert gt_depth and pred disp image tensor to valid point-matched numpy array
    input:
        gt_depth: gt depth image tensor, [B*W*H]
        pred_disp: disparity image tensor from network, [B*C*W*H],C=1. Normalized by width

    output:
    """
    # convert tensor to numpy
    gt = gt_depth.cpu().numpy()
    _,h,w = gt_depth.size()
    pred_disp = F.interpolate(pred_disp,size=[h,w],mode='nearest')
    pred_disp = pred_disp.squeeze(1).cpu().numpy()
    
    # get valid point
    mask = (gt>0)
    gt = gt[mask]
    pred_disp = pred_disp[mask]

    # convert disp to depth
    pred_depth = disp2depth(pred_disp,w)
    pred = pred_depth
    return gt,pred

def compute_errors(gt, pred):
    """
    gt: depth gt numpy array, orginat image size
    pred: depth prediction image numpy array.
    """

    ### from Godard. 
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# def compute_errors_tensor(gt, pred):
#     ### compute error with tensor

#     # get valid value from gt and pred tensor
#     mask = (gt>0)
#     gt = gt[mask]
#     pred = pred[mask]

#     thresh = torch.maximum((gt/pred), (pred/gt))
#     a1 = (thresh < 1.25   ).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     rmse = (gt - pred)**2
#     rmse = torch.sqrt(rmse)

#     abs_rel = (torch.abs((gt-pred) / gt)).mean()

#     sq_rel = (((gt - pred)**2) / gt).mean()
