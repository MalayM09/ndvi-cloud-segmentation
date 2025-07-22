import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_file_pairs(ndvi_folder, gt_folder):
    ndvi_folder = Path(ndvi_folder)
    gt_folder = Path(gt_folder)

    ndvi_files = sorted(ndvi_folder.glob('*.tif'))
    file_pairs = []

    for ndvi_file in ndvi_files:
        ndvi_num = ndvi_file.stem.split('_')[1]
        gt_file = gt_folder / f"GT_CM1_{ndvi_num}.tif"
        if gt_file.exists():
            file_pairs.append((ndvi_file, gt_file))

    return file_pairs

def create_train_val_split(file_pairs, val_ratio=0.2, seed=42):
    train_pairs, val_pairs = train_test_split(
        file_pairs, test_size=val_ratio, random_state=seed
    )
    return train_pairs, val_pairs

def generate_patches(file_pairs, preprocessor, patch_size=256, stride=128):
    training_data = []

    for ndvi_file, gt_file in file_pairs:
        with rasterio.open(ndvi_file) as src:
            ndvi = src.read(1)

        with rasterio.open(gt_file) as src:
            gt_mask = src.read(1)

        h, w = ndvi.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                ndvi_patch = ndvi[i:i+patch_size, j:j+patch_size]
                mask_patch = gt_mask[i:i+patch_size, j:j+patch_size]

                cloud_pixels = np.sum(mask_patch == 1)
                shadow_pixels = np.sum(mask_patch == 2)
                total_pixels = patch_size * patch_size
                meaningful = cloud_pixels + shadow_pixels

                if meaningful > total_pixels * 0.01 or cloud_pixels > 100 or shadow_pixels > 100 or len(np.unique(mask_patch)) > 1:
                    try:
                        processed = preprocessor.preprocess_single_image_array(ndvi_patch, apply_augmentation=True)
                        training_data.append({
                            'ndvi_sam': processed['sam_input'],
                            'gt_mask': mask_patch.astype(np.uint8),
                            'source_file': ndvi_file.stem,
                            'patch_coords': (i, j),
                            'cloud_pixels': cloud_pixels,
                            'shadow_pixels': shadow_pixels
                        })
                    except Exception:
                        continue

    return training_data
