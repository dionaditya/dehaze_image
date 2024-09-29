#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn 
import math
from pytorch_msssim import ssim
from torchvision.models import efficientnet_b0
from pytorch_msssim import ms_ssim


# Load dataset paths
train_hazy_path = './ohazy/hazy/'
val_hazy_path = './ohazy/val/hazy'
val_gt_path = './ohazy/val/gt'
gt_path = './ohazy/gt/'
savepath = ""
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cut an image uniformly into 9 256x256 patches
def cut_patches(img):
    patches = []
    width = 0
    height = 0

    for i in range(2):
        width = 0
        if i == 1:
            height = img.shape[0] - 256

        for j in range(3):
            if j == 1:
                width = (img.shape[1] // 2) - 128
            elif j == 2:
                width = img.shape[1] - 256

            cut = img[height:height+256, width:width+256]
            patches.append(cut)

    return patches

# Custom Dataset for loading images


def display_images(inputs, targets, outputs, epoch):
    # Convert tensors to numpy arrays for visualization
    def tensor_to_numpy(tensor):
        tensor = tensor.cpu().detach()
        tensor = tensor.permute(0, 2, 3, 1).numpy()  # Change dimensions for plotting
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        return tensor

    inputs_np = tensor_to_numpy(inputs)
    targets_np = tensor_to_numpy(targets)
    outputs_np = tensor_to_numpy(outputs)

    # Display images
    for i in range(min(inputs_np.shape[0], 4)):  # Limit to first 4 images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(inputs_np[i])
        axes[0].set_title('Input Image')
        axes[1].imshow(targets_np[i])
        axes[1].set_title('Target Image')
        axes[2].imshow(outputs_np[i])
        axes[2].set_title('Model Output')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()  # Display in Colab notebook
        
class HazyClearDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir=None, transform=None, is_test=False):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.is_test = is_test

        # List hazy images
        self.hazy_images = self.list_images(hazy_dir, '')

        if not is_test:
            if clear_dir is None:
                raise ValueError("clear_dir must be provided when is_test is False")
            self.clear_images = self.list_images(clear_dir, '')
            # Match hazy and clear images by filename (ignoring extension)
            self.image_pairs = self.match_images()
        else:
            self.image_pairs = [(img, None) for img in self.hazy_images]

    def list_images(self, directory, suffix):
        extensions = ('.jpg', '.jpeg', '.png', '.JPEG')
        return [f for f in os.listdir(directory) if f.lower().endswith(extensions) and suffix in f]

    def match_images(self):
        image_pairs = []
        hazy_dict = {os.path.splitext(f)[0].replace('', ''): f for f in self.hazy_images}
        for clear_img in self.clear_images:
            clear_name = os.path.splitext(clear_img)[0].replace('', '')
            if clear_name in hazy_dict:
                image_pairs.append((clear_img, hazy_dict[clear_name]))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clear_img, hazy_img = self.image_pairs[idx]
        hazy_img_path = os.path.join(self.hazy_dir, hazy_img)

        hazy_image = Image.open(hazy_img_path).convert("RGB")

        if clear_img is not None:
            clear_img_path = os.path.join(self.clear_dir, clear_img)
            clear_image = Image.open(clear_img_path).convert("RGB")


            if self.transform:
                hazy_image = self.transform(hazy_image)
                clear_image = self.transform(clear_image)
            
            return hazy_image, clear_image
        else:
            if self.transform:
                hazy_image = self.transform(hazy_image)
            
            return hazy_image
        
# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# DataLoader
train_dataset = HazyClearDataset(hazy_dir=train_hazy_path, clear_dir=gt_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_hazy_path = './dataset/test/val'

# Create test dataset and loader
test_dataset = HazyClearDataset(hazy_dir=val_hazy_path, clear_dir=val_gt_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Set shuffle=False for test data


class ESPCN(nn.Module):
    def __init__(self, num_channel, scale):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 64, (5, 5), padding=5//2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding= 3//2)
        self.conv3 = nn.Conv2d(32, scale**2, (3, 3), padding=3//2)
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.weight_init()

    def weight_init(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv2.bias)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixelshuffle(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (H*W, B, C)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        attn_output, _ = self.attention(q, k, v)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # Reshape back to (B, C, H, W)
        return attn_output

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.global_avg_pool(x).view(b, c)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1, 1)
        return x * se

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class HighResDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with SE block"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(HighResDoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class HighResDown(nn.Module):
    """Downscaling with maxpool then high-resolution double conv"""

    def __init__(self, in_channels, out_channels):
        super(HighResDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            HighResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class HighResUp(nn.Module):
    """Upscaling then high-resolution double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(HighResUp, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = HighResDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = HighResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (H*W, B, C)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        attn_output, _ = self.attention(q, k, v)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # Reshape back to (B, C, H, W)
        return attn_output

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class LightweightTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, num_heads=4, num_layers=4, patch_size=4, upscale_factor=2):
        super(LightweightTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.patch_embed(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, -1, h // self.patch_embed.proj.kernel_size[0], w // self.patch_embed.proj.kernel_size[0])
        x = self.upsample(x)
        return x
    
class GlobalFeatureTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, num_heads=4, num_layers=16, patch_size=4):
        super(GlobalFeatureTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.patch_embed(x)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(b, -1, h // self.patch_embed.proj.kernel_size[0], w // self.patch_embed.proj.kernel_size[0])
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)  # Ensure same spatial dimensions as input
        x = self.proj(x)
        return x
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)  # Added more depth for higher resolution
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec5 = DoubleConv(1024 + 512, 512)  # Adjusted for added depth
        self.dec4 = DoubleConv(512 + 256, 256)
        self.dec3 = DoubleConv(256 + 128, 128)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        dec5 = self.up(enc5)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.dec5(dec5)
        
        dec4 = self.up(dec5)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up(dec2)
        dec1 = self.dec1(dec1)
        
        return dec1
    
class SuperResolutionTransformer(nn.Module):
    def __init__(self, dim=256, num_heads=8, num_layers=6):
        super(SuperResolutionTransformer, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.embedding = nn.Linear(256*256, dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 256*256, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_linear = nn.Linear(dim, 256*256)
        
    def forward(self, x):
        # Assuming x shape is [batch_size, 1, 256, 256]
        batch_size = x.size(0)
        
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 256*256]
        x = self.embedding(x)  # [batch_size, 256*256, dim]
        x = x + self.positional_encoding  # Add positional encoding
        
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, dim]
        x = self.transformer(x)  # Pass through transformer layers
        x = x.permute(1, 0, 2)  # Back to [batch_size, 256*256, dim]
        
        x = self.output_linear(x)  # [batch_size, 256*256, 256*256]
        x = x.view(batch_size, 1, 256, 256)  # Reshape to [batch_size, 1, 256, 256]
        
        return x
    
class SRTransformerUNet(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96, num_heads=4, num_layers=4, patch_size=4, upscale_factor=2):
        super(SRTransformerUNet, self).__init__()
        self.sr_transformer = LightweightTransformer(in_channels, embed_dim, num_heads, num_layers, patch_size, upscale_factor)
        self.unet = UNet()
        self.global_transformer = GlobalFeatureTransformer(in_channels, embed_dim, num_heads, num_layers, patch_size)
    def forward(self, x):
        sr = self.sr_transformer(x)
        unet_features = self.unet(sr)
        global_features = self.global_transformer(unet_features)
        
        return unet_features + global_features


def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded. Resuming from epoch {epoch+1} with best validation loss {best_val_loss}")
        return model, optimizer, epoch, best_val_loss
    else:
        print(f"No checkpoint found at '{filepath}'. Starting from scratch.")
        return model, optimizer, 0, float('inf')

def display_images(inputs, targets, outputs, epoch):
    # Convert tensors to numpy arrays for visualization
    def tensor_to_numpy(tensor):
        tensor = tensor.cpu().detach()
        tensor = tensor.permute(0, 2, 3, 1).numpy()  # Change dimensions for plotting
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        return tensor

    inputs_np = tensor_to_numpy(inputs)
    targets_np = tensor_to_numpy(targets)
    outputs_np = tensor_to_numpy(outputs)

    # Display images
    for i in range(min(inputs_np.shape[0], 4)):  # Limit to first 4 images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(inputs_np[i])
        axes[0].set_title('Input Image')
        axes[1].imshow(targets_np[i])
        axes[1].set_title('Target Image')
        axes[2].imshow(outputs_np[i])
        axes[2].set_title('Model Output')
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()  # Display in Colab notebook


def combined_loss(outputs, targets, alpha=0.84):
    mse_loss = nn.MSELoss()(outputs, targets)
    
    # Calculate MS-SSIM loss
    ms_ssim_loss = 1 - ms_ssim(outputs, targets, data_range=255, size_average=True)
    
    return alpha * mse_loss + (1 - alpha) * ms_ssim_loss

if __name__ == '__main__':
    # Parameters
    n_channels = 3  # Input channels (RGB)
    n_classes = 3   # Output channels (RGB)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRTransformerUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Load from checkpoint if available
    checkpoint_path = os.path.join(savepath, "current_checkpoint.pth")
    model, optimizer, start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

    # Training loop
    num_epochs = 1000
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            clear_images_resized = F.interpolate(targets, size=outputs.shape[2:])
            loss = combined_loss(outputs, clear_images_resized)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                clear_images_resized = F.interpolate(targets, size=outputs.shape[2:])
                loss = combined_loss(outputs, clear_images_resized)
                val_loss += loss.item()
        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

        # Check if the current validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}). Saving model...")
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, "best_checkpoint.pth")
            
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
        }, f"current_checkpoint.pth")

    print("Training complete.")