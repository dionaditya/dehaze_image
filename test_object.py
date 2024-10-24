import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tensorflow_addons as tfa
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential, save_model

# Load the model without needing to specify custom objects
loaded_model = tf.keras.models.load_model("vit_object_detector")

loaded_model.summary()

# Load the dataset
ds = load_dataset("keremberke/license-plate-object-detection", name="full")
image_size = 256


# Define IoU function
def bounding_box_intersection_over_union(box_predicted, box_truth):
    top_x_intersect = max(box_predicted[0], box_truth[0])
    top_y_intersect = max(box_predicted[1], box_truth[1])
    bottom_x_intersect = min(box_predicted[2], box_truth[2])
    bottom_y_intersect = min(box_predicted[3], box_truth[3])

    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(0, bottom_y_intersect - top_y_intersect + 1)
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (box_predicted[3] - box_predicted[1] + 1)
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (box_truth[3] - box_truth[1] + 1)

    return intersection_area / float(box_predicted_area + box_truth_area - intersection_area)

# Evaluate the model
import matplotlib.patches as patches

def test_data_generator(dataset):
    x_test = []
    y_test = []

    for i in range(len(dataset)):
        example = dataset[i]
        
        # Load image
        img_array = np.array(example['image'])
        print(f"Image at index {i} loaded with shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        if img_array is None or img_array.size == 0:
            print(f"Image at index {i} is empty or not loaded properly.")
            continue  # Skip this iteration

        # Check original image range
        print(f"Original image min: {img_array.min()}, max: {img_array.max()}")

        # Get original image dimensions
        original_height, original_width = img_array.shape[:2]

        # Normalize and resize the image
        img_array = img_array / 255.0
        img_array = cv2.resize(img_array, (image_size, image_size))
        
        # Check the processed image shape and type
        print(f"Processed image {i}: shape={img_array.shape}, dtype={img_array.dtype}")

        # Get the bounding box and scale it based on the original image dimensions
        bbox = example['objects']['bbox'][0]
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

        # Scale bounding box based on original image size
        x_min_scaled = x_min / original_width
        y_min_scaled = y_min / original_height
        x_max_scaled = x_max / original_width
        y_max_scaled = y_max / original_height

        # Normalize bounding box based on resized image dimensions (0 to 1 range)
        bbox_normalized = [
            x_min_scaled,
            y_min_scaled,
            x_max_scaled,
            y_max_scaled
        ]
        
        x_test.append(img_array)
        y_test.append(bbox_normalized)

    return np.array(x_test), np.array(y_test)
# Generate x_test and y_test
x_test, y_test = test_data_generator(ds['test'])  # You can use any batch size

# Evaluate the model as before
mean_iou = 0  # Initialize mean IoU

for i, input_image in enumerate(x_test[:50]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    im = input_image

    # Display images
    ax1.imshow(im, cmap='gray' if im.ndim == 2 else None)
    ax2.imshow(im, cmap='gray' if im.ndim == 2 else None)

    input_image = cv2.resize(input_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    input_image = np.expand_dims(input_image, axis=0)

    preds = loaded_model.predict(input_image)
    print(preds)
    preds = preds[0]
    (h, w) = im.shape[0:2]
    
    top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)
    bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    
    print(box_predicted)

    # Draw predicted bounding box
    rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    # Draw ground truth bounding box
    gt_box = y_test[i]
    
    rect_gt = patches.Rectangle((gt_box[0] * w, gt_box[1] * h), (gt_box[2] - gt_box[0]) * w, (gt_box[3] - gt_box[1]) * h,
                                 linewidth=2, edgecolor='g', facecolor='none')
    ax1.add_patch(rect_gt)

    iou = bounding_box_intersection_over_union(box_predicted, gt_box)
    mean_iou += iou

plt.show()

# Final mean IoU
mean_iou /= len(x_test[:10])
print(f'Mean IoU: {mean_iou}')