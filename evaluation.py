import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from evaluation_utils import compute_errors
from evaluation_utils import disp2depth
from evaluation_utils import get_depth_array

def evaluate_with_gt(gt_depth,pred_disp):
    """
    evaluate disparity with ground truth
    gt_depth: gt depth image tensor
    pred: disparity image tensor(0-1)
    """
    gt, pred = get_depth_array(gt_depth,pred_disp)
    
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt, pred)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    