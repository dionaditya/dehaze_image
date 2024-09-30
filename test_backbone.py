import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from unet_backbone import SRTransformerUNet  # Adjust this import as necessary
from PIL import Image
import time
from skimage.metrics import structural_similarity
import math
import cv2

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(im1, im2):
    original = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2GRAY)
    contrast = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2GRAY)
    score, _ = structural_similarity(original, contrast, full=True)
    return score

def display_images(inputs, outputs, save_dir="output_images", prefix="image"):
    def tensor_to_numpy(tensor):
        tensor = tensor.cpu().detach()
        tensor = tensor.permute(0, 2, 3, 1).numpy()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return tensor

    inputs_np = tensor_to_numpy(inputs)
    outputs_np = tensor_to_numpy(outputs)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(min(inputs_np.shape[0], 1)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(inputs_np[i])
        axes[0].set_title('Input Image')
        axes[1].imshow(outputs_np[i])
        axes[1].set_title('Output Image')
        
        input_image_path = os.path.join(save_dir, f"{prefix}_input_{i}.png")
        output_image_path = os.path.join(save_dir, f"{prefix}_output_{i}.png")
        
        plt.imsave(input_image_path, inputs_np[i])
        plt.imsave(output_image_path, outputs_np[i])
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()
        plt.close(fig)

def visualize_output(model, image_path, transform, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        output = model(image)
        
        output_image_tensor = output.squeeze(0).cpu()
        output_image_tensor = output_image_tensor.clamp(0, 1)
        
        output_image = transforms.ToPILImage()(output_image_tensor)
        output_image_path = './output_images/test.png'
        output_image.save(output_image_path)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to run the model: {elapsed_time} seconds")
    
    return image.squeeze(0).cpu(), output_image_tensor

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

def compute_metrics_for_folder(model, hazy_folder_path, gt_folder_path, transform, device='cuda'):
    psnr_values = []
    ssim_values = []

    hazy_files = [f for f in os.listdir(hazy_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    gt_files = [f for f in os.listdir(gt_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    hazy_filenames = {os.path.splitext(f)[0]: f for f in hazy_files}
    gt_filenames = {os.path.splitext(f)[0]: f for f in gt_files}

    common_filenames = set(hazy_filenames.keys()).intersection(set(gt_filenames.keys()))

    for base_filename in common_filenames:
        hazy_image_path = os.path.join(hazy_folder_path, hazy_filenames[base_filename])
        gt_image_path = os.path.join(gt_folder_path, gt_filenames[base_filename])

        input_image, output_image = visualize_output(model, hazy_image_path, transform, device)

        gt_image = Image.open(gt_image_path).convert('RGB')
        gt_image = transform(gt_image).to(device).cpu()

        input_image = transforms.ToPILImage()(input_image)
        output_image = transforms.ToPILImage()(output_image)
        gt_image = transforms.ToPILImage()(gt_image)

        output_image = output_image.resize((1024, 1024), Image.BICUBIC)
        
        psnr_value = psnr(np.array(gt_image), np.array(output_image))
        ssim_value = ssim(gt_image, output_image)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

        print(f"Image: {base_filename} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    hazy_folder_path = 'ohazy/val/hazy'  # Path to your folder with hazy images
    gt_folder_path = 'ohazy/val/gt'  # Path to your folder with ground truth images

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SRTransformerUNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    model, optimizer, start_epoch, best_val_loss = load_checkpoint("./backbone_current_checkpoint.pth", model, optimizer)

    compute_metrics_for_folder(model, hazy_folder_path, gt_folder_path, transform, device)
