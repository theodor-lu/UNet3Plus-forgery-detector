#!/usr/bin/env python3
"""
Script to delete images and masks with:
1. More than 50% coverage (mask_area / image_area)
2. Probe mask area differs from donor mask area by more than 30%
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
COVERAGE_THRESHOLD = 0.5  # 50% coverage threshold
AREA_DIFF_THRESHOLD = 0.3  # 30% difference threshold between donor and probe mask areas

def calculate_mask_coverage(image_path, donor_mask_folder, probe_mask_folder):
    """
    Calculate the mask coverage percentage and area difference for an image using original resolution.
    
    Args:
        image_path: Path to the image file
        donor_mask_folder: Folder containing donor masks
        probe_mask_folder: Folder containing probe masks
    
    Returns:
        coverage: Coverage percentage (0.0 to 1.0)
        mask_area: Total area of the combined mask (in pixels)
        donor_area: Area of donor mask (in pixels)
        probe_area: Area of probe mask (in pixels)
        area_diff: Percentage difference between donor and probe areas (0.0 to 1.0+)
    """
    filename = os.path.basename(image_path)
    
    # Load original image to get its dimensions
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Get original image dimensions (height, width)
    if len(img.shape) == 2:
        original_height, original_width = img.shape
    else:
        original_height, original_width = img.shape[:2]
    
    original_size = (original_width, original_height)  # cv2.resize expects (width, height)
    
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
    
    # Calculate individual mask areas (count of mask pixels)
    donor_area = (donor_mask > 127).sum()
    probe_area = (probe_mask > 127).sum()
    
    # Combine masks (union operation)
    combined_mask = np.maximum(donor_mask, probe_mask)
    
    # Calculate combined area
    mask_area = (combined_mask > 127).sum()
    
    # Calculate coverage percentage using original image dimensions
    total_pixels = original_height * original_width
    coverage = mask_area / total_pixels if total_pixels > 0 else 0.0
    
    # Calculate percentage difference between donor and probe areas
    # Use the larger area as denominator to get relative difference
    max_area = max(donor_area, probe_area, 1)  # Use 1 to avoid division by zero
    area_diff = abs(probe_area - donor_area) / max_area
    
    return coverage, mask_area, donor_area, probe_area, area_diff

def load_image_and_mask(image_path, donor_mask_folder, probe_mask_folder):
    """
    Load image and combined mask for visualization using original resolution.
    
    Args:
        image_path: Path to the image file
        donor_mask_folder: Folder containing donor masks
        probe_mask_folder: Folder containing probe masks
    
    Returns:
        image: Image array (H, W, C) in RGB format, normalized to [0, 1]
        combined_mask: Combined mask array (H, W) normalized to [0, 1]
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
    
    # Combine masks (union operation)
    combined_mask = np.maximum(donor_mask, probe_mask)
    combined_mask = combined_mask.astype(np.float32) / 255.0
    
    return img, combined_mask

def load_image_and_masks_separate(image_path, donor_mask_folder, probe_mask_folder):
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

def display_images_with_overlays(images_to_delete, num_images=10):
    """
    Display images with mask overlays.
    
    Args:
        images_to_delete: List of image info dictionaries
        num_images: Number of images to display
    """
    num_images = min(num_images, len(images_to_delete))
    
    # Create figure with subplots (4 columns: original, donor, probe, overlay)
    fig, axes = plt.subplots(num_images, 4, figsize=(20, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        item = images_to_delete[i]
        filename = item['filename']
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            # Load image and masks separately
            img, donor_mask, probe_mask = load_image_and_masks_separate(
                image_path, DONOR_MASK_FOLDER, PROBE_MASK_FOLDER
            )
            
            # Original image
            axes[i, 0].imshow(np.clip(img, 0, 1))
            title = f'{filename}\nCoverage: {item["coverage"]*100:.2f}%\nArea Diff: {item["area_diff"]*100:.2f}%'
            axes[i, 0].set_title(title, fontsize=8)
            axes[i, 0].axis('off')
            
            # Donor mask (red)
            donor_colored = np.zeros_like(img)
            donor_colored[:, :, 0] = donor_mask
            axes[i, 1].imshow(np.clip(donor_colored, 0, 1))
            axes[i, 1].set_title(f'Donor (Red)\nArea: {item["donor_area"]}', fontsize=8)
            axes[i, 1].axis('off')
            
            # Probe mask (blue)
            probe_colored = np.zeros_like(img)
            probe_colored[:, :, 2] = probe_mask
            axes[i, 2].imshow(np.clip(probe_colored, 0, 1))
            axes[i, 2].set_title(f'Probe (Blue)\nArea: {item["probe_area"]}', fontsize=8)
            axes[i, 2].axis('off')
            
            # Overlay: original image with both masks
            overlay = img.copy() * 0.6
            # Add donor mask in red
            donor_overlay = np.zeros_like(overlay)
            donor_overlay[:, :, 0] = donor_mask
            overlay = overlay + donor_overlay * 0.4
            # Add probe mask in blue
            probe_overlay = np.zeros_like(overlay)
            probe_overlay[:, :, 2] = probe_mask
            overlay = overlay + probe_overlay * 0.4
            overlay = np.clip(overlay, 0, 1)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(f'Overlay\n{item["reason"]}', fontsize=8)
            axes[i, 3].axis('off')
            
        except Exception as e:
            axes[i, 0].text(0.5, 0.5, f'Error loading\n{filename}\n{str(e)}', 
                           ha='center', va='center', transform=axes[i, 0].transAxes, fontsize=8)
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')
            axes[i, 2].axis('off')
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization to file instead of displaying
    output_file = 'high_coverage_preview.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_file}")

def main():
    print("=" * 60)
    print("Delete Images and Masks")
    print(f"  - Coverage >{COVERAGE_THRESHOLD*100:.0f}%")
    print(f"  - Area difference >{AREA_DIFF_THRESHOLD*100:.0f}%")
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
    print("Calculating mask coverage...")
    
    # Calculate coverage and area difference for all images
    images_to_delete = []
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        coverage, mask_area, donor_area, probe_area, area_diff = calculate_mask_coverage(
            image_path, DONOR_MASK_FOLDER, PROBE_MASK_FOLDER
        )
        
        # Check if image should be deleted (either high coverage OR large area difference)
        should_delete = False
        reason = []
        
        if coverage > COVERAGE_THRESHOLD:
            should_delete = True
            reason.append(f"coverage>{COVERAGE_THRESHOLD*100:.0f}%")
        
        if area_diff > AREA_DIFF_THRESHOLD:
            should_delete = True
            reason.append(f"area_diff>{AREA_DIFF_THRESHOLD*100:.0f}%")
        
        if should_delete:
            images_to_delete.append({
                'filename': filename,
                'coverage': coverage,
                'mask_area': mask_area,
                'donor_area': donor_area,
                'probe_area': probe_area,
                'area_diff': area_diff,
                'reason': ' + '.join(reason)
            })
    
    # Sort by coverage (descending)
    images_to_delete.sort(key=lambda x: x['coverage'], reverse=True)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"Found {len(images_to_delete)} images to delete")
    print(f"  - Coverage >{COVERAGE_THRESHOLD*100:.0f}%: {sum(1 for x in images_to_delete if x['coverage'] > COVERAGE_THRESHOLD)}")
    print(f"  - Area diff >{AREA_DIFF_THRESHOLD*100:.0f}%: {sum(1 for x in images_to_delete if x['area_diff'] > AREA_DIFF_THRESHOLD)}")
    print("=" * 60)
    
    if len(images_to_delete) == 0:
        print("No images to delete!")
        return
    
    # Show preview of images to be deleted
    print(f"\nPreview (first 10):")
    print(f"{'Filename':<40} {'Coverage':<10} {'Area Diff':<10} {'Donor':<10} {'Probe':<10} {'Reason':<20}")
    print("-" * 100)
    for item in images_to_delete[:10]:
        print(f"{item['filename']:<40} {item['coverage']*100:>8.2f}% {item['area_diff']*100:>8.2f}% "
              f"{item['donor_area']:>10} {item['probe_area']:>10} {item['reason']:<20}")
    
    if len(images_to_delete) > 10:
        print(f"... and {len(images_to_delete) - 10} more")
    
    # Display first 10 images with mask overlays
    print("\nDisplaying first 10 images with mask overlays...")
    display_images_with_overlays(images_to_delete, num_images=10)
    
    # Ask for confirmation
    print("\n" + "=" * 60)
    response = input(f"Delete {len(images_to_delete)} images and their masks? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Deletion cancelled.")
        return
    
    # Delete images and masks
    print("\nDeleting files...")
    deleted_count = 0
    error_count = 0
    
    for item in tqdm(images_to_delete, desc="Deleting"):
        filename = item['filename']
        
        # Delete image
        image_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")
            error_count += 1
            continue
        
        # Delete donor mask
        donor_mask_path = os.path.join(DONOR_MASK_FOLDER, filename)
        try:
            if os.path.exists(donor_mask_path):
                os.remove(donor_mask_path)
        except Exception as e:
            print(f"Error deleting {donor_mask_path}: {e}")
        
        # Delete probe mask
        probe_mask_filename = os.path.splitext(filename)[0] + '.jpg'
        probe_mask_path = os.path.join(PROBE_MASK_FOLDER, probe_mask_filename)
        try:
            if os.path.exists(probe_mask_path):
                os.remove(probe_mask_path)
        except Exception as e:
            print(f"Error deleting {probe_mask_path}: {e}")
        
        deleted_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Deletion Summary")
    print("=" * 60)
    print(f"Images deleted: {deleted_count}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")
    
    # Save list of deleted files
    deleted_file = 'deleted_images.txt'
    with open(deleted_file, 'w') as f:
        f.write(f"Deleted Images\n")
        f.write("=" * 100 + "\n")
        f.write(f"Criteria:\n")
        f.write(f"  - Coverage >{COVERAGE_THRESHOLD*100:.0f}%\n")
        f.write(f"  - Area difference >{AREA_DIFF_THRESHOLD*100:.0f}%\n")
        f.write(f"Total deleted: {len(images_to_delete)}\n")
        f.write(f"{'Filename':<40} {'Coverage':<10} {'Area Diff':<10} {'Donor':<10} {'Probe':<10} {'Reason':<20}\n")
        f.write("-" * 100 + "\n")
        for item in images_to_delete:
            f.write(f"{item['filename']:<40} {item['coverage']*100:>8.2f}% {item['area_diff']*100:>8.2f}% "
                    f"{item['donor_area']:>10} {item['probe_area']:>10} {item['reason']:<20}\n")
    
    print(f"\nList of deleted files saved to: {deleted_file}")

if __name__ == "__main__":
    main()

