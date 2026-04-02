import math
import torch
import torch.nn as nn
import timm

from models.grasp_model import GraspModel
from models.vit_grasp import GRConvDecoder  # reuse decoder
from models.ttt_block import TTT  


class ViTBlockWithTTT(nn.Module):
    """
    Drop-in-ish replacement for timm's ViT Block:
      x = x + attn(norm1(x))
      x = x + mlp(norm2(x))

    We replace attn(...) with TTT(..., h, w).
    """
    def __init__(self, vit_block: nn.Module, dim: int, num_heads: int, use_conv_branch: bool = True):
        super().__init__()

        # Copy LayerNorm + MLP + DropPath from the original timm block
        self.norm1 = vit_block.norm1
        self.norm2 = vit_block.norm2
        self.mlp = vit_block.mlp

        # timm may have one droppath module, or separate; handle both patterns
        self.drop_path1 = getattr(vit_block, "drop_path1", None)
        self.drop_path2 = getattr(vit_block, "drop_path2", None)
        if self.drop_path1 is None or self.drop_path2 is None:
            dp = getattr(vit_block, "drop_path", nn.Identity())
            self.drop_path1 = dp
            self.drop_path2 = dp

        # New TTT module (this is what replaces attention)
        self.ttt = TTT(dim=dim, num_heads=num_heads, qkv_bias=True, use_conv_branch=use_conv_branch)

    def forward(self, x: torch.Tensor, h: int, w: int):
        x = x + self.drop_path1(self.ttt(self.norm1(x), h=h, w=w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViTTTTGraspModel(GraspModel):
    """
    Same as ViTGraspModel but uses TTT in place of attention in every ViT block.
    Patch-only tokens (no CLS) so TTT can reshape tokens into h x w.
    """
    def __init__(
        self,
        input_channels: int = 3,
        vit_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        decoder_base_ch: int = 128,
        decoder_dropout_p: float = 0.0,
        use_conv_branch: bool = True,
        image_size: int = 224,
    ):
        super().__init__()

        self.in_adapter = nn.Identity() if input_channels == 3 else nn.Conv2d(input_channels, 3, kernel_size=1)

        # Build timm ViT (reuse its patch_embed/pos_embed/norm/mlp weights)
        self.vit = timm.create_model(vit_name, pretrained=pretrained, img_size=image_size)

        dim = self.vit.embed_dim

        # Replace blocks
        new_blocks = []
        for blk in self.vit.blocks:
            # timm attention has num_heads here:
            nh = blk.attn.num_heads
            new_blocks.append(ViTBlockWithTTT(blk, dim=dim, num_heads=nh, use_conv_branch=use_conv_branch))
        self.blocks = nn.ModuleList(new_blocks)

        self.decoder = GRConvDecoder(
            in_ch=dim,
            base_ch=decoder_base_ch,
            dropout_p=decoder_dropout_p,
            head_kernel=1,
            output_size=image_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.in_adapter(x)

        # Patch-only ViT forward (no CLS)
        x = self.vit.patch_embed(x)                 # (B, 196, C)
        x = x + self.vit.pos_embed[:, 1:, :]        # skip CLS pos-embed
        x = self.vit.pos_drop(x)

        B, N, C = x.shape
        H = W = int(math.isqrt(N))
        assert H * W == N, f"Tokens N={N} not a square; cannot reshape to grid for TTT."

        for blk in self.blocks:
            x = blk(x, h=H, w=W)

        x = self.vit.norm(x)                        # (B, 196, C)

        feat = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, 14, 14)
        return self.decoder(feat)