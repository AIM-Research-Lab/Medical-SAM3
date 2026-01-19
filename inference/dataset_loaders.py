"""
Dataset loaders for medical image segmentation datasets.
Provides unified interface for loading images and ground truth masks.
"""

import os
from pathlib import Path
from typing import Iterator, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image


# Base data directory - update this path to your local data location
DATA_ROOT = Path("../medsam_data")

# Text prompts for each dataset
DATASET_PROMPTS = {
    "CHASE_DB1": "Retinal Blood Vessel",
    "STARE": "Retinal Blood Vessel",
    "CVC-ClinicDB": "Polyp",
    "ETIS-Larib": "Polyp",
    "PH2": "Skin Lesion",
}


@dataclass
class Sample:
    """A single dataset sample."""
    image: np.ndarray          # RGB image (H, W, 3)
    gt_mask: np.ndarray        # Binary mask (H, W)
    dataset_name: str
    sample_id: str
    text_prompt: str


def load_chase_db1(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load CHASE_DB1 dataset.

    Structure:
        CHASE_DB1/
        ├── Image_XXY.jpg        # Original image
        ├── Image_XXY_1stHO.png  # Expert 1 annotation
        └── Image_XXY_2ndHO.png  # Expert 2 annotation
    """
    dataset_dir = DATA_ROOT / "CHASE_DB1"

    # Find all original images
    image_files = sorted([f for f in os.listdir(dataset_dir)
                          if f.endswith('.jpg')])

    if max_samples:
        image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = img_file.replace('.jpg', '')

        # Load image
        img_path = dataset_dir / img_file
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth (use 1st expert annotation)
        mask_file = f"{sample_id}_1stHO.png"
        mask_path = dataset_dir / mask_file

        if not mask_path.exists():
            continue

        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="CHASE_DB1",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["CHASE_DB1"]
        )


def load_stare(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load STARE dataset.

    Structure:
        STARE/
        ├── imXXXX.ppm       # Original image (decompressed)
        ├── imXXXX.ah.ppm    # AH expert annotation
        └── imXXXX.vk.ppm    # VK expert annotation
    """
    dataset_dir = DATA_ROOT / "STARE"

    # Find all original images (not .ah or .vk)
    image_files = sorted([f for f in os.listdir(dataset_dir)
                          if f.endswith('.ppm') and '.ah.' not in f and '.vk.' not in f])

    if max_samples:
        image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = img_file.replace('.ppm', '')

        # Load image
        img_path = dataset_dir / img_file
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth (use AH expert annotation)
        mask_file = f"{sample_id}.ah.ppm"
        mask_path = dataset_dir / mask_file

        if not mask_path.exists():
            continue

        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="STARE",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["STARE"]
        )


def load_cvc_clinicdb(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load CVC-ClinicDB dataset.

    Structure:
        CVC-ClinicDB/PNG/
        ├── Original/N.png      # Original image
        └── Ground Truth/N.png  # Mask
    """
    dataset_dir = DATA_ROOT / "CVC-ClinicDB" / "PNG"
    original_dir = dataset_dir / "Original"
    gt_dir = dataset_dir / "Ground Truth"

    # Find all images
    image_files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')],
                         key=lambda x: int(x.replace('.png', '')))

    if max_samples:
        image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = img_file.replace('.png', '')

        # Load image
        img_path = original_dir / img_file
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth
        mask_path = gt_dir / img_file

        if not mask_path.exists():
            continue

        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="CVC-ClinicDB",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["CVC-ClinicDB"]
        )


def load_etis_larib(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load ETIS-Larib dataset.

    Structure:
        ETIS-Larib/
        ├── images/N.png   # Original image
        └── masks/N.png    # Mask
    """
    dataset_dir = DATA_ROOT / "ETIS-Larib"
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    # Find all images
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')],
                         key=lambda x: int(x.replace('.png', '')))

    if max_samples:
        image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = img_file.replace('.png', '')

        # Load image
        img_path = image_dir / img_file
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth
        mask_path = mask_dir / img_file

        if not mask_path.exists():
            continue

        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="ETIS-Larib",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["ETIS-Larib"]
        )


def load_ph2(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load PH2 dataset.

    Structure:
        PH2/PH2Dataset/PH2 Dataset images/IMDXXX/
        ├── IMDXXX_Dermoscopic_Image/IMDXXX.bmp  # Original image
        └── IMDXXX_lesion/IMDXXX_lesion.bmp      # Mask
    """
    dataset_dir = DATA_ROOT / "PH2" / "PH2Dataset" / "PH2 Dataset images"

    # Find all sample directories
    sample_dirs = sorted([d for d in os.listdir(dataset_dir)
                          if os.path.isdir(dataset_dir / d) and d.startswith('IMD')])

    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    for sample_id in sample_dirs:
        sample_dir = dataset_dir / sample_id

        # Load image
        img_path = sample_dir / f"{sample_id}_Dermoscopic_Image" / f"{sample_id}.bmp"

        if not img_path.exists():
            continue

        image = np.array(Image.open(img_path).convert('RGB'))

        # Load ground truth
        mask_path = sample_dir / f"{sample_id}_lesion" / f"{sample_id}_lesion.bmp"

        if not mask_path.exists():
            continue

        gt_mask = np.array(Image.open(mask_path).convert('L'))
        gt_mask = (gt_mask > 127).astype(np.uint8)  # Binarize

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="PH2",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["PH2"]
        )


# Dataset loader registry
DATASET_LOADERS = {
    "CHASE_DB1": load_chase_db1,
    "STARE": load_stare,
    "CVC-ClinicDB": load_cvc_clinicdb,
    "ETIS-Larib": load_etis_larib,
    "PH2": load_ph2,
}


def load_dataset(dataset_name: str, max_samples: Optional[int] = None) -> Iterator[Sample]:
    """Load a dataset by name."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[dataset_name](max_samples)


def load_all_datasets(max_samples_per_dataset: Optional[int] = None) -> Iterator[Sample]:
    """Load all datasets."""
    for dataset_name in DATASET_LOADERS:
        yield from load_dataset(dataset_name, max_samples_per_dataset)


def get_dataset_info() -> dict:
    """Get information about available datasets."""
    info = {}
    for name, loader in DATASET_LOADERS.items():
        samples = list(loader())
        info[name] = {
            "num_samples": len(samples),
            "text_prompt": DATASET_PROMPTS[name],
        }
        if samples:
            info[name]["image_shape"] = samples[0].image.shape
    return info


if __name__ == "__main__":
    # Test loading
    print("Testing dataset loaders...")
    print("=" * 60)

    for dataset_name in DATASET_LOADERS:
        print(f"\n{dataset_name}:")
        try:
            samples = list(load_dataset(dataset_name, max_samples=2))
            print(f"  Loaded {len(samples)} samples")
            if samples:
                s = samples[0]
                print(f"  Image shape: {s.image.shape}")
                print(f"  Mask shape: {s.gt_mask.shape}")
                print(f"  Mask unique values: {np.unique(s.gt_mask)}")
                print(f"  Text prompt: {s.text_prompt}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Dataset loading test complete!")
