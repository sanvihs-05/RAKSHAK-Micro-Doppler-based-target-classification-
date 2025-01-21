import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
base_path = r"C:\Users\sanvi\OneDrive\Desktop\DIAT-uSAT_dataset\DIAT-uSAT_dataset"
output_base_path = r"C:\Users\sanvi\OneDrive\Desktop\Resized_Normalized_Dataset"

# Fixed size for ResNet50
target_size = (224, 224)

# Ensure output directory exists
os.makedirs(output_base_path, exist_ok=True)

# Categories
categories = ['3_long_blades_rotor', '3_short_blade_rotor_1', '3_short_blade_rotor_2', 
              'Bird', 'Bird+mini-helicopter_1', 'Bird+mini-helicopter_2', 
              'drone_1', 'drone_2', 'RC plane_1', 'RC plane_2']

# Data and labels
data = []    # Resized, normalized image data
labels = []  # Corresponding category labels

# Resize, normalize, and prepare images
for category in categories:
    folder = os.path.join(base_path, category)
    output_folder = os.path.join(output_base_path, category)
    os.makedirs(output_folder, exist_ok=True)  # Create output folder for each category
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            # Load and resize the image
            img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
            img_resized = img.resize(target_size)
            
            # Normalize the image (scale pixel values to range [0, 1])
            img_normalized = np.array(img_resized) / 255.0
            
            # Append data and labels
            data.append(img_normalized)
            labels.append(category)
            output_img_path = os.path.join(output_folder, img_name)
            img_resized.save(output_img_path)
            
            print(f"{category} -> {img_name} processed and saved.")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
# Convert data and labels to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)
print(f"Total images processed: {len(data)}")
# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example: Augment a single image
example_image = data[0]  # Select the first image
augmented_image = datagen.random_transform(example_image)
