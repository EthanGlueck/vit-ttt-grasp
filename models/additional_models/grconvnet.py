import torch.nn as nn
import torch.nn.functional as F

from models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=1, dropout=False, prob=0.0, channel_size=32):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)
        self.res5 = ResidualBlock(128, 128)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=2)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=2)
        self.width_output = nn.Conv2d(32, 1, kernel_size=2)

        self.dropout1 = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        pos_output = self.pos_output(self.dropout1(x))
        cos_output = self.cos_output(self.dropout1(x))
        sin_output = self.sin_output(self.dropout1(x))
        width_output = self.width_output(self.dropout1(x))

        return pos_output, cos_output, sin_output, width_output

    def flops(self, H=224, W=224, input_channels=1):
        """
        Calculate FLOPs for GenerativeResnet with input size HxW.
        """
        flops = 0
        
        # conv1: Conv2d(input_channels, 32, 9, stride=1, padding=4)
        out_h1 = (H + 2*4 - 9) // 1 + 1
        out_w1 = (W + 2*4 - 9) // 1 + 1
        flops += out_h1 * out_w1 * input_channels * 32 * 9 * 9
        
        # conv2: Conv2d(32, 64, 4, stride=2, padding=1)
        out_h2 = (out_h1 + 2*1 - 4) // 2 + 1
        out_w2 = (out_w1 + 2*1 - 4) // 2 + 1
        flops += out_h2 * out_w2 * 32 * 64 * 4 * 4
        
        # conv3: Conv2d(64, 128, 4, stride=2, padding=1)
        out_h3 = (out_h2 + 2*1 - 4) // 2 + 1
        out_w3 = (out_w2 + 2*1 - 4) // 2 + 1
        flops += out_h3 * out_w3 * 64 * 128 * 4 * 4
        
        # res1-res5: 5 ResidualBlocks, each with 2 Conv2d(128, 128, 3, padding=1)
        flops += 5 * 2 * (out_h3 * out_w3 * 128 * 128 * 3 * 3)
        
        # conv4: ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1)
        out_h4 = (out_h3 - 1) * 2 - 2*1 + 4 + 1
        out_w4 = (out_w3 - 1) * 2 - 2*1 + 4 + 1
        flops += out_h4 * out_w4 * 128 * 64 * 4 * 4
        
        # conv5: ConvTranspose2d(64, 32, 4, stride=2, padding=2, output_padding=1)
        out_h5 = (out_h4 - 1) * 2 - 2*2 + 4 + 1
        out_w5 = (out_w4 - 1) * 2 - 2*2 + 4 + 1
        flops += out_h5 * out_w5 * 64 * 32 * 4 * 4
        
        # conv6: ConvTranspose2d(32, 32, 9, stride=1, padding=4)
        out_h6 = (out_h5 - 1) * 1 - 2*4 + 9
        out_w6 = (out_w5 - 1) * 1 - 2*4 + 9
        flops += out_h6 * out_w6 * 32 * 32 * 9 * 9
        
        # Output convs: 4 x Conv2d(32, 1, 2)
        out_h7 = out_h6 - 2 + 1
        out_w7 = out_w6 - 2 + 1
        flops += 4 * out_h7 * out_w7 * 32 * 1 * 2 * 2
        
        return int(flops)
