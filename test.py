import torch
from datasets import load_dataset
from object_detection import CustomDataset, model
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ds = load_dataset("keremberke/license-plate-object-detection", name="full")

train_ds = CustomDataset(ds['test'], image_size=256    )

# Create a subset with 200 items
subset_indices = list(range(10))  # Get the indices for the first 200 items
train_subset = Subset(train_ds, subset_indices)

test_loader = DataLoader(train_subset, batch_size=16, shuffle=False)

model.load_state_dict(torch.load("trained_vit_license_plate_detection.pth"))
def visualize_images(data_loader, model):
    model.eval()  # Ensure model is in evaluation mode
    images, labels, class_tensors = next(iter(data_loader))  # Get a batch of images and labels
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(images)  # Get bounding box predictions

    # Convert tensors to numpy for visualization
    images = images.permute(0, 2, 3, 1).numpy()  # Change from [N, C, H, W] to [N, H, W, C]
    outputs = outputs.numpy()  # Convert output tensor to numpy
    labels = labels.numpy()  # Convert label tensor to numpy

    # Plot the images with bounding boxes
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    
    for img, ax, output, label in zip(images, axes, outputs, labels):
        ax.imshow(img)  # Show the image
        ax.axis('off')  # Hide axes
        
        print(output.size)
        if output.size == 4:
            x_min, y_min, x_max, y_max = output
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            linewidth=2, edgecolor='r', facecolor='none')
        print(label, label.size)
        if label.size == 4:
            x_min, y_min, x_max, y_max = label[0]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                            linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)
    plt.tight_layout()
    plt.show()

# Visualize images from the test dataset
visualize_images(test_loader, model)