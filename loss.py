import torch
import torch.nn as nn
import torch.nn.functional as F

class DispLoss(nn.modules.Module):
    """
    upsample multi-scale disp to image size to computing reconstruction loss 
    """
    def __init__(self, n=4, SSIM_w=0.65, disp_gradient_w=0.1, lr_w=0.4,ssim_kernel=3):
        super(DispLoss, self).__init__()
        self.SSIM_w = SSIM_w
        #self.match_w = match_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n
        self.ssim_kernel = ssim_kernel  #add ssim kernel size parameter. dafault as 3

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros',align_corners=True)

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(self.ssim_kernel, 1)(x)
        mu_y = nn.AvgPool2d(self.ssim_kernel, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(self.ssim_kernel, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(self.ssim_kernel, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(self.ssim_kernel, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, image):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = self.gradient_x(image)
        image_gradients_y = self.gradient_y(image)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,
                     keepdim=True)) 
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,
                     keepdim=True))

        smoothness_x = [disp_gradients_x[i] * weights_x
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def forward(self, dispL,dispR, LRimg):
        """
        Args:
            disp: [disp1, disp2, disp3, disp4]
            LRimg: [left, right]

        Return:
            (float): The loss
        """
        left_img, right_img = LRimg

        _,_,height,width = left_img.size()

        disp_left_est = []
        disp_right_est = []
        for i in range(self.n):
            disp_left_est.append(F.interpolate(dispL[i],[height,width],mode='bilinear',align_corners=True)) 
            disp_right_est.append(F.interpolate(dispR[i],[height,width],mode='bilinear',align_corners=True)) 

        # Prepare disparities
        # disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in disp_upsample]
        # disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in disp_upsample]
        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        # Generate images
        left_est = [self.generate_image_left(right_img,
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_img,
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_img)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_img)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_img))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_img)) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_img)) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_img)) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** (self.n-i-1)
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** (self.n-i-1)
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss
        #loss = image_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        return loss

class DispLoss_LR(nn.modules.Module):
    """
    upsample multi-scale disp to image size to computing depth reconstruction loss
    update: add edge smooth loss
    """
    def __init__(self, disp_gradient_w=0.1,n=4):
        super(DispLoss_LR, self).__init__()
        self.n = n
        self.disp_gradient_w = disp_gradient_w

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros',align_corners=True)

        return output

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy
    
    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def disp_smoothness(self, disp, img):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,
                     keepdim=True)) 
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,
                     keepdim=True)) 

        smoothness_x = [disp_gradients_x[i] * weights_x
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]
    
    def forward(self, disps_L, disps_R,img_L):
        """
        Args:
            disp: [disp1, disp2, disp3, disp4]
            disp[i]: (N,C,H,W)

        Return:
            (float): The loss

        note: only update left ir image depth smooth now
        """
        _,_,height,width = disps_L[self.n-1].size()

        disp_left_est = []
        for i in range(self.n-1):
            disp_left_est.append(F.interpolate(disps_L[i],[height,width],mode='bilinear',align_corners=True)) 
        disp_left_est.append(disps_L[self.n-1])

        disp_right_est = []
        for i in range(self.n-1):
            disp_right_est.append(F.interpolate(disps_R[i],[height,width],mode='bilinear',align_corners=True)) 
        disp_right_est.append(disps_R[self.n-1])

        # Prepare disparities
        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    img_L)
        # disp_right_smoothness = self.disp_smoothness(disp_right_est,
        #                                              img_R)

        # Disparities smoothness
        smooth_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** (self.n-i-1)
                          for i in range(self.n)]
        # disp_right_loss = [torch.mean(torch.abs(
        #                    disp_right_smoothness[i])) / 2 ** i
        #                    for i in range(self.n)]
        disp_gradient_loss = sum(smooth_left_loss)

        loss = lr_loss + self.disp_gradient_w * disp_gradient_loss
        return loss

class Disp_consistency(nn.modules.Module):

    def __init__(self,n=4):
        super(Disp_consistency,self).__init__()
        self.n = n
    
    def forward(self,disp1,disp2):
        
        loss = [torch.mean(torch.abs(disp1[i]
                    - disp2[i])) for i in range(self.n)]
        loss = sum(loss)
        return loss