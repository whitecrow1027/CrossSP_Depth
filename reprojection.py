import torch
import torch.nn as nn
import torch.nn.functional as F


class Reprojection(nn.modules.Module):
    """
    reprojection image with disp
    """
    def __init__(self):
        super(Reprojection, self).__init__()

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

    def forward(self, image,disp,mode):
        """
        mode: left, right
        """
        if mode == 'left':
            warpimg = self.generate_image_left(image,disp)
        if mode == 'right':
            warpimg = self.generate_image_right(image,disp)

        return warpimg