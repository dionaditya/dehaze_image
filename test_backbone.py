import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from unet_backbone import SRTransformerUNet, HazyClearDataset
from PIL import Image
import time 

def display_images(inputs, outputs, save_dir="output_images", prefix="image"):
    # Convert tensors to numpy arrays for visualization
    def tensor_to_numpy(tensor):
        tensor = tensor.cpu().detach()
        tensor = tensor.permute(0, 2, 3, 1).numpy()  # Change dimensions for plotting
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        return tensor

    inputs_np = tensor_to_numpy(inputs)
    outputs_np = tensor_to_numpy(outputs)

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Display and save images
    for i in range(min(inputs_np.shape[0], 1)):  # Limit to the first image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(inputs_np[i])
        axes[0].set_title('Input Image')
        axes[1].imshow(outputs_np[i])
        axes[1].set_title('Output Image')
        
           # Save individual images
        input_image_path = os.path.join(save_dir, f"{prefix}_input_{i}.png")
        output_image_path = os.path.join(save_dir, f"{prefix}_output_{i}.png")
        
        plt.imsave(input_image_path, inputs_np[i])
        plt.imsave(output_image_path, outputs_np[i])
        
        for ax in axes:
            ax.axis('off')
        
        plt.show()  # Display in notebook or script

     


        # Optionally, close the figure after displaying to free memory
        plt.close(fig)


def visualize_output(model, image_path, transform, device='cuda'):
    """
    Visualize the model output for a single image.
    
    Parameters:
        model (nn.Module): The model to use for prediction
        image_path (str): Path to the image to be tested
        transform (callable): Transform to apply to the image
        device (str): Device to run the model on
    """
    model.eval()
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    def tensor_to_numpy(tensor):
        tensor = tensor.cpu().detach()
        tensor = tensor.permute(0, 2, 3, 1).numpy()  # Change dimensions for plotting
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
        return tensor

    with torch.no_grad():
        start_time = time.time()
        output = model(image)
        
         # Display image and output
        # Convert the tensor back to an image
        output_image_tensor = output.squeeze(0).cpu()  # Remove batch dimension and move to CPU
        output_image_tensor = output_image_tensor.clamp(0, 1)
        
        output_image = transforms.ToPILImage()(output_image_tensor)

        # Save the image
        output_image_path = './output_images/test.png'
        output_image.save(output_image_path)
        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken to run the model: {elapsed_time} seconds")
    
    display_images(image, output)

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
          
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),  # Adjust the size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_image_path = 'dataset/test/hazy/images.png'  # Path to your single test image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SRTransformerUNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Load the state dictionary
    model, optimizer, start_epoch, best_val_loss = load_checkpoint("./current_checkpoint.pth", model, optimizer)
    
    visualize_output(model, test_image_path, transform, device)
