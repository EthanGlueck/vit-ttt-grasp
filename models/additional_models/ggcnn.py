import torch.nn as nn
import torch.nn.functional as F

filter_sizes = [32, 16, 8, 8, 16, 32]
kernel_sizes = [9, 5, 3, 3, 5, 9]
strides = [3, 2, 2, 2, 2, 3]


class GGCNN(nn.Module):
    """
    GG-CNN
    Equivalient to the Keras Model used in the RSS Paper (https://arxiv.org/abs/1804.05172)
    """
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def flops(self, H=224, W=224):
        """
        Calculate FLOPs for GGCNN with input size HxW.
        Assumes input_channels=1.
        """
        flops = 0
        
        # conv1: Conv2d(1, 32, 9, stride=3, padding=3)
        out_h1 = (H + 2*3 - 9) // 3 + 1
        out_w1 = (W + 2*3 - 9) // 3 + 1
        flops += out_h1 * out_w1 * 1 * 32 * 9 * 9
        
        # conv2: Conv2d(32, 16, 5, stride=2, padding=2)
        out_h2 = (out_h1 + 2*2 - 5) // 2 + 1
        out_w2 = (out_w1 + 2*2 - 5) // 2 + 1
        flops += out_h2 * out_w2 * 32 * 16 * 5 * 5
        
        # conv3: Conv2d(16, 8, 3, stride=2, padding=1)
        out_h3 = (out_h2 + 2*1 - 3) // 2 + 1
        out_w3 = (out_w2 + 2*1 - 3) // 2 + 1
        flops += out_h3 * out_w3 * 16 * 8 * 3 * 3
        
        # convt1: ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1)
        out_h4 = (out_h3 - 1) * 2 - 2*1 + 3 + 1
        out_w4 = (out_w3 - 1) * 2 - 2*1 + 3 + 1
        flops += out_h4 * out_w4 * 8 * 8 * 3 * 3
        
        # convt2: ConvTranspose2d(8, 16, 5, stride=2, padding=2, output_padding=1)
        out_h5 = (out_h4 - 1) * 2 - 2*2 + 5 + 1
        out_w5 = (out_w4 - 1) * 2 - 2*2 + 5 + 1
        flops += out_h5 * out_w5 * 8 * 16 * 5 * 5
        
        # convt3: ConvTranspose2d(16, 32, 9, stride=3, padding=3, output_padding=1)
        out_h6 = (out_h5 - 1) * 3 - 2*3 + 9 + 1
        out_w6 = (out_w5 - 1) * 3 - 2*3 + 9 + 1
        flops += out_h6 * out_w6 * 16 * 32 * 9 * 9
        
        # Output convs: 4 x Conv2d(32, 1, 2)
        out_h7 = out_h6 - 2 + 1
        out_w7 = out_w6 - 2 + 1
        flops += 4 * out_h7 * out_w7 * 32 * 1 * 2 * 2
        
        return int(flops)

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        # GGCNN expects 300x300 input, but the tfgrasp dataset provides 224x224
        # giving a 244x244 image in returns a 248x248, so we resize the output to match
        H, W = y_pos.shape[-2], y_pos.shape[-1]
        if pos_pred.shape[-2:] != (H, W):
            pos_pred   = F.interpolate(pos_pred,   size=(H, W), mode='bilinear', align_corners=False)
            cos_pred   = F.interpolate(cos_pred,   size=(H, W), mode='bilinear', align_corners=False)
            sin_pred   = F.interpolate(sin_pred,   size=(H, W), mode='bilinear', align_corners=False)
            width_pred = F.interpolate(width_pred, size=(H, W), mode='bilinear', align_corners=False)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
