import numpy as np
import rasterio
from pathlib import Path

def preprocess_cm1_masks(cm1_folder, output_folder):
    cm1_folder = Path(cm1_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    class_mapping = {
        0: 0,  # Clear
        1: 1,  # Thick clouds
        2: 0,  # Thin clouds (treated as clear)
        3: 2   # Cloud shadows
    }

    cm1_files = list(cm1_folder.glob('*.tif'))

    class_statistics = {0: 0, 1: 0, 2: 0, 3: 0}
    new_class_statistics = {0: 0, 1: 0, 2: 0}
    processed_count = 0

    for cm1_file in cm1_files:
        try:
            with rasterio.open(cm1_file) as src:
                cm1_data = src.read(1)
                profile = src.profile.copy()

            unique_values, counts = np.unique(cm1_data, return_counts=True)
            for val, count in zip(unique_values, counts):
                if val in class_statistics:
                    class_statistics[val] += count

            ground_truth = np.zeros_like(cm1_data, dtype=np.uint8)
            for old_class, new_class in class_mapping.items():
                mask = (cm1_data == old_class)
                ground_truth[mask] = new_class
                new_class_statistics[new_class] += np.sum(mask)

            output_filename = f"GT_{cm1_file.stem}.tif"
            output_path = output_folder / output_filename

            profile.update({
                'dtype': 'uint8',
                'count': 1,
                'compress': 'lzw'
            })

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(ground_truth, 1)

            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(cm1_files)} files...")

        except Exception as e:
            print(f"Error processing {cm1_file.name}: {e}")

    total_original = sum(class_statistics.values())
    print("Original Class Distribution:")
    for class_id, count in class_statistics.items():
        pct = (count / total_original) * 100
        print(f"  Class {class_id}: {count} pixels ({pct:.1f}%)")

    total_new = sum(new_class_statistics.values())
    print("New Class Distribution:")
    for class_id, count in new_class_statistics.items():
        pct = (count / total_new) * 100
        print(f"  Class {class_id}: {count} pixels ({pct:.1f}%)")

    return new_class_statistics
