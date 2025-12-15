#!/usr/bin/env python3
"""
Script to find and output the names of the 50 easiest images.
Easiest images are defined as those with the largest mask areas.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display required)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dataset paths (same as in config)
IMAGE_FOLDER = 'archive/copymove_img/img'
DONOR_MASK_FOLDER = 'archive/copymove_annotations/donor_mask'
PROBE_MASK_FOLDER = 'archive/copymove_annotations/probe_mask'
IMG_SIZE = (224, 224)  # Same as config.IMG_SIZE

def calculate_mask_area(image_path, donor_mask_folder, probe_mask_folder, img_size):
    """
    Calculate the combined mask area for an image.
    
    Args:
        image_path: Path to the image file
        donor_mask_folder: Folder containing donor masks
        probe_mask_folder: Folder containing probe masks
        img_size: Target size for resizing masks
    
    Returns:
        mask_area: Total area of the combined mask (in pixels)
    """
    filename = os.path.basename(image_path)
    
    # Load donor mask
    donor_mask_path = os.path.join(donor_mask_folder, filename)
    donor_mask = cv2.imread(donor_mask_path, cv2.IMREAD_GRAYSCALE)
    if donor_mask is None:
        donor_mask = np.zeros(img_size, dtype=np.uint8)
    else:
        donor_mask = cv2.resize(donor_mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    # Load probe mask
    probe_mask_filename = os.path.splitext(filename)[0] + '.jpg'
    probe_mask_path = os.path.join(probe_mask_folder, probe_mask_filename)
    probe_mask = cv2.imread(probe_mask_path, cv2.IMREAD_GRAYSCALE)
    if probe_mask is None:
        probe_mask = np.zeros(img_size, dtype=np.uint8)
    else:
        probe_mask = cv2.resize(probe_mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    # Combine masks (union operation)
    combined_mask = np.maximum(donor_mask, probe_mask)
    
    # Calculate area (count of mask pixels, not sum of values)
    # Count pixels where mask value > 127 (threshold for binary/grayscale masks)
    area = (combined_mask > 127).sum()
    
    return area

def load_image_and_masks(image_path, donor_mask_folder, probe_mask_folder):
    """
    Load image and separate masks for visualization using original resolution.
    
    Args:
        image_path: Path to the image file
        donor_mask_folder: Folder containing donor masks
        probe_mask_folder: Folder containing probe masks
    
    Returns:
        image: Image array (H, W, C) in RGB format, normalized to [0, 1]
        donor_mask: Donor mask array (H, W) normalized to [0, 1]
        probe_mask: Probe mask array (H, W) normalized to [0, 1]
    """
    filename = os.path.basename(image_path)
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Handle different image formats
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original image dimensions
    original_height, original_width = img.shape[:2]
    original_size = (original_width, original_height)  # cv2.resize expects (width, height)
    
    # Normalize image to [0, 1] (keep original size)
    img = img.astype(np.float32) / 255.0
    
    # Load donor mask
    donor_mask_path = os.path.join(donor_mask_folder, filename)
    donor_mask = cv2.imread(donor_mask_path, cv2.IMREAD_GRAYSCALE)
    if donor_mask is None:
        donor_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    else:
        # Resize mask to match original image size
        donor_mask = cv2.resize(donor_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Load probe mask
    probe_mask_filename = os.path.splitext(filename)[0] + '.jpg'
    probe_mask_path = os.path.join(probe_mask_folder, probe_mask_filename)
    probe_mask = cv2.imread(probe_mask_path, cv2.IMREAD_GRAYSCALE)
    if probe_mask is None:
        probe_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    else:
        # Resize mask to match original image size
        probe_mask = cv2.resize(probe_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Normalize masks to [0, 1]
    donor_mask = donor_mask.astype(np.float32) / 255.0
    probe_mask = probe_mask.astype(np.float32) / 255.0
    
    return img, donor_mask, probe_mask

def save_images_with_overlays(top_images, num_images=20):
    """
    Save images with mask overlays showing donor and probe masks in different colors.
    
    Args:
        top_images: List of image info dictionaries
        num_images: Number of images to save
    """
    num_images = min(num_images, len(top_images))
    
    # Create figure with subplots: 4 columns (original, donor, probe, overlay)
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for masks: donor = red, probe = blue
    DONOR_COLOR = np.array([1.0, 0.0, 0.0])  # Red
    PROBE_COLOR = np.array([0.0, 0.0, 1.0])  # Blue
    
    for i in range(num_images):
        item = top_images[i]
        filename = item['filename']
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            # Load image and masks
            img, donor_mask, probe_mask = load_image_and_masks(
                image_path, DONOR_MASK_FOLDER, PROBE_MASK_FOLDER
            )
            
            # Original image
            axes[i, 0].imshow(np.clip(img, 0, 1))
            axes[i, 0].set_title(f'{filename}\nRank: {i+1}, Area: {item["area"]} px')
            axes[i, 0].axis('off')
            
            # Donor mask (red)
            donor_colored = np.zeros_like(img)
            donor_colored[:, :, 0] = donor_mask  # Red channel
            axes[i, 1].imshow(np.clip(donor_colored, 0, 1))
            axes[i, 1].set_title('Donor Mask (Red)')
            axes[i, 1].axis('off')
            
            # Probe mask (blue)
            probe_colored = np.zeros_like(img)
            probe_colored[:, :, 2] = probe_mask  # Blue channel
            axes[i, 2].imshow(np.clip(probe_colored, 0, 1))
            axes[i, 2].set_title('Probe Mask (Blue)')
            axes[i, 2].axis('off')
            
            # Overlay: original image with both masks
            # Start with original image at reduced opacity
            overlay = img.copy() * 0.6
            
            # Add donor mask in red
            donor_overlay = np.zeros_like(overlay)
            donor_overlay[:, :, 0] = donor_mask  # Red channel
            overlay = overlay + donor_overlay * 0.4
            
            # Add probe mask in blue
            probe_overlay = np.zeros_like(overlay)
            probe_overlay[:, :, 2] = probe_mask  # Blue channel
            overlay = overlay + probe_overlay * 0.4
            
            # Where both masks overlap, it will show purple (red + blue)
            overlay = np.clip(overlay, 0, 1)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Red=Donor, Blue=Probe)')
            axes[i, 3].axis('off')
            
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error loading\n{filename}\n{str(e)}', 
                           ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization to file
    output_file = 'easiest_20_images_with_overlays.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")

def main():
    print("=" * 60)
    print("Finding 50 Easiest Images (Largest Mask Areas)")
    print("=" * 60)
    
    # Get all image files
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder not found: {IMAGE_FOLDER}")
        return
    
    image_files = sorted([
        f for f in os.listdir(IMAGE_FOLDER) 
        if f.endswith('.tif')
    ])
    
    print(f"\nFound {len(image_files)} images")
    print("Calculating mask areas...")
    
    # Calculate mask areas for all images
    image_areas = []
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        area = calculate_mask_area(image_path, DONOR_MASK_FOLDER, PROBE_MASK_FOLDER, IMG_SIZE)
        image_areas.append({
            'filename': filename,
            'area': area
        })
    
    # Sort by area (descending - largest = easiest)
    image_areas.sort(key=lambda x: x['area'], reverse=True)
    
    # Get top 50 easiest images
    top_50 = image_areas[:50]
    
    # Output results
    print("\n" + "=" * 60)
    print("Top 50 Easiest Images (Largest Mask Areas)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Filename':<50} {'Mask Area':<12}")
    print("-" * 60)
    
    for rank, item in enumerate(top_50, 1):
        print(f"{rank:<6} {item['filename']:<50} {item['area']:<12}")
    
    # Save to file
    output_file = 'easiest_50_images.txt'
    with open(output_file, 'w') as f:
        f.write("Top 50 Easiest Images (Largest Mask Areas)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Filename':<50} {'Mask Area':<12}\n")
        f.write("-" * 60 + "\n")
        for rank, item in enumerate(top_50, 1):
            f.write(f"{rank:<6} {item['filename']:<50} {item['area']:<12}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Save visualization of top 20 images with mask overlays
    print("\nSaving visualization of top 20 images with mask overlays...")
    save_images_with_overlays(top_50, num_images=20)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total images processed: {len(image_areas)}")
    print(f"Largest mask area: {image_areas[0]['area']} pixels")
    print(f"Smallest mask area (in top 50): {top_50[-1]['area']} pixels")
    print(f"Average mask area (top 50): {np.mean([x['area'] for x in top_50]):.1f} pixels")
    print(f"Median mask area (top 50): {np.median([x['area'] for x in top_50]):.1f} pixels")

if __name__ == "__main__":
    main()

