import torch
import torch.nn as nn
import timm
from models.grasp_model import GraspModel


class GRConvDecoder(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 128, dropout_p: float = 0.0, head_kernel: int = 1,
                 output_size: int = 224):
        super().__init__()
        self.output_size = output_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch // 8),
            nn.ReLU(inplace=True),
        )

        head_ch = base_ch // 8

        self.conv6 = nn.Conv2d(head_ch, head_ch, kernel_size=9, stride=1, padding=4)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        pad = 0 if head_kernel == 1 else 1
        self.q_head   = nn.Conv2d(head_ch, 1, kernel_size=head_kernel, padding=pad)
        self.cos_head = nn.Conv2d(head_ch, 1, kernel_size=head_kernel, padding=pad)
        self.sin_head = nn.Conv2d(head_ch, 1, kernel_size=head_kernel, padding=pad)
        self.w_head   = nn.Conv2d(head_ch, 1, kernel_size=head_kernel, padding=pad)

        self.head_kernel = head_kernel

    def forward(self, feat: torch.Tensor):
        x = self.proj(feat)
        x = self.up(x)
        x = self.conv6(x)
        x = self.dropout(x)

        q   = self.q_head(x)
        cos = self.cos_head(x)
        sin = self.sin_head(x)
        w   = self.w_head(x)

        if self.head_kernel == 2:
            q = q[..., :self.output_size, :self.output_size]
            cos = cos[..., :self.output_size, :self.output_size]
            sin = sin[..., :self.output_size, :self.output_size]
            w = w[..., :self.output_size, :self.output_size]

        return q, cos, sin, w


class ViTGraspModel(GraspModel):
    """
    Pipeline-compatible model:
      input: (B, input_channels, 224, 224)
      output: (pos, cos, sin, width) each (B,1,224,224)
    """
    def __init__(
        self,
        input_channels: int = 3,
        vit_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        decoder_base_ch: int = 128,
        decoder_dropout_p: float = 0.0,
        image_size: int = 224,
    ):
        super().__init__()

        # Adapter: map {1,3,4} channels -> 3 channels expected by pretrained ViT
        self.in_adapter = nn.Identity() if input_channels == 3 else nn.Conv2d(input_channels, 3, kernel_size=1)

        self.vit = timm.create_model(vit_name, pretrained=pretrained, img_size=image_size)
        embed_dim = self.vit.embed_dim

        self.decoder = GRConvDecoder(
            in_ch=embed_dim,
            base_ch=decoder_base_ch,
            dropout_p=decoder_dropout_p,
            head_kernel=1,
            output_size=image_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.in_adapter(x)

       # ---- Patch-only ViT forward (no CLS) ----
        x = self.vit.patch_embed(x)                 # (B, 196, C) for 224x224, patch16
        # pos_embed is (1, 197, C) = [CLS + patches], so skip the first token
        x = x + self.vit.pos_embed[:, 1:, :]
        x = self.vit.pos_drop(x)

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)                        # (B, 196, C)
        patch = x                                   # (B, N, C)

        B, N, C = patch.shape
        H = W = int(N ** 0.5)
        feat = patch.transpose(1, 2).reshape(B, C, H, W)  # (B, C, 14, 14)

        # Return same format as GenerativeResnet: pos, cos, sin, width
        return self.decoder(feat)