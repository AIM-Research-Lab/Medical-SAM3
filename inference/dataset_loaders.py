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
    "TN3K": "thyroid nodule",    
    "DDTI": "thyroid nodule", 
    "TG3K": "thyroid nodule",  
    "HC18": "fetal head",
    "DSB18": "cell nucleus",
    "CoNSeP": "cell nucleus",
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
        
def load_hc18(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load HC18 dataset (Fetal Head).
    Structure: Single folder containing both images and annotations.
    Naming: 
       Image: XXX_HC.png
       Mask:  XXX_HC_Annotation.png
    """
    dataset_dir = DATA_ROOT / "HC18"
    
    if not dataset_dir.exists():
        print(f"ERROR: HC18 dir not found at {dataset_dir}")
        return
    all_files = sorted([f for f in os.listdir(dataset_dir) if f.lower().endswith('.png')])
    image_files = [f for f in all_files if 'Annotation' not in f]
    
    if max_samples:
        image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = os.path.splitext(img_file)[0]
        img_path = dataset_dir / img_file
        mask_file = img_file.replace('.png', '_Annotation.png')
        mask_path = dataset_dir / mask_file
        
        if not mask_path.exists():
            mask_path = dataset_dir / img_file.replace('.png', '_Annotation.png')
            if not mask_path.exists():
                continue

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)

            yield Sample(image, gt_mask, "HC18", sample_id, DATASET_PROMPTS["HC18"])
        except Exception as e:
            print(f"Error loading HC18 {sample_id}: {e}")


def load_ddti(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load DDTI dataset.

    Structure:
        DDTI dataset/
        ├── image/*.png  # Original images
        └── mask/*.png   # Corresponding masks
    """
    dataset_dir = DATA_ROOT / "DDTI"
    image_dir = dataset_dir / "image"
    mask_dir = dataset_dir / "mask"

    if not image_dir.exists(): return

    # Find all images
    image_files = sorted([f for f in os.listdir(image_dir) 
                          if f.endswith('.PNG') or f.endswith('.png')])
    
    # Try to sort numerically if filenames contain digits
    try:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    except:
        pass

    if max_samples: image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = os.path.splitext(img_file)[0]
        img_path = image_dir / img_file
        
        # Try several possible mask filenames (case-insensitive extension)
        mask_path = None
        for name in [img_file, img_file.replace('.PNG', '.png'), img_file.replace('.png', '.PNG')]:
            if (mask_dir / name).exists():
                mask_path = mask_dir / name
                break
        
        if not mask_path: continue

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)
            yield Sample(image, gt_mask, "DDTI", sample_id, DATASET_PROMPTS["DDTI"])
        except Exception as e:
            print(f"Error loading DDTI {sample_id}: {e}")


def load_tn3k(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load TN3K dataset.

    Structure:
        tn3k/
        ├── test-image/*.jpg      # Test images
        ├── test-mask/*.png/jpg   # Test masks
        ├── trainval-image/*.jpg  # Train/val images
        └── trainval-mask/*.png/jpg  # Train/val masks
    """
    dataset_dir = DATA_ROOT / "TN3K"
    # Pair image and mask folders
    folder_pairs = [("test-image", "test-mask"), ("trainval-image", "trainval-mask")]
    count = 0
    
    for img_subdir, mask_subdir in folder_pairs:
        image_dir = dataset_dir / img_subdir
        mask_dir = dataset_dir / mask_subdir
        if not image_dir.exists(): continue

        # Find all jpg images
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])

        for img_file in image_files:
            if max_samples and count >= max_samples: return
            
            sample_id = os.path.splitext(img_file)[0]
            img_path = image_dir / img_file
            mask_path = mask_dir / img_file 
            
            # Try png if jpg mask is not found
            if not mask_path.exists():
                 mask_path = mask_dir / img_file.replace('.jpg', '.png')
            
            if not mask_path.exists(): continue

            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                gt_mask = np.array(Image.open(mask_path).convert('L'))
                gt_mask = (gt_mask > 127).astype(np.uint8)
                yield Sample(image, gt_mask, "TN3K", sample_id, DATASET_PROMPTS["TN3K"])
                count += 1
            except: continue


def load_tg3k(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load TG3K dataset.

    Structure:
        tg3k/
        ├── image/*.jpg  # Original images
        └── mask/*.png/jpg  # Corresponding masks
    """
    dataset_dir = DATA_ROOT / "TG3K"
    image_dir = dataset_dir / "image"
    mask_dir = dataset_dir / "mask"

    if not image_dir.exists(): return []

    # Find all jpg images
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    if max_samples: image_files = image_files[:max_samples]

    for img_file in image_files:
        sample_id = os.path.splitext(img_file)[0]
        img_path = image_dir / img_file
        mask_path = mask_dir / img_file
        
        # Try png mask if jpg mask is not found
        if not mask_path.exists():
            mask_path = mask_dir / img_file.replace('.jpg', '.png')
        
        if not mask_path.exists(): continue

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
            gt_mask = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_mask > 127).astype(np.uint8)
            yield Sample(image, gt_mask, "TG3K", sample_id, DATASET_PROMPTS["TG3K"])
        except: continue

def load_consep(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load CoNSeP dataset.
    Structure:
        CoNSeP/
        ├── Train/
        │   ├── Images/*.png
        │   └── Labels/*.mat
        └── Test/
            ├── Images/*.png
            └── Labels/*.mat
            
    Note: Both Train and Test sets are used for evaluation as requested.
    Labels are .mat files containing 'inst_map' which we convert to binary mask.
    """
    dataset_dir = DATA_ROOT / "CoNSeP"
    subsets = ["Test", "Train"]
    
    count = 0
    
    for subset in subsets:
        subset_dir = dataset_dir / subset
        image_dir = subset_dir / "Images"
        label_dir = subset_dir / "Labels"
        
        if not image_dir.exists():
            continue

        # Find all png images
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        for img_file in image_files:
            if max_samples and count >= max_samples:
                return

            sample_id = os.path.splitext(img_file)[0]
            
            # Load Image
            img_path = image_dir / img_file
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
                continue

            # Load Label (.mat)
            mat_file = f"{sample_id}.mat"
            mat_path = label_dir / mat_file
            
            if not mat_path.exists():
                continue
                
            try:
                mat_data = scipy.io.loadmat(str(mat_path))
                if 'inst_map' in mat_data:
                    inst_map = mat_data['inst_map']
                    gt_mask = (inst_map > 0).astype(np.uint8)
                elif 'type_map' in mat_data:
                    gt_mask = (mat_data['type_map'] > 0).astype(np.uint8)
                else:
                    print(f"Warning: No 'inst_map' or 'type_map' found in {mat_file}")
                    continue

                yield Sample(
                    image=image,
                    gt_mask=gt_mask,
                    dataset_name="CoNSeP",
                    sample_id=f"{subset}_{sample_id}",
                    text_prompt=DATASET_PROMPTS["CoNSeP"]
                )
                count += 1
                
            except Exception as e:
                print(f"Error loading CoNSeP mat file {sample_id}: {e}")

def load_dsb18(max_samples: Optional[int] = None) -> Iterator[Sample]:
    """
    Load Data Science Bowl 2018 (DSB18) dataset.
    
    Structure based on your screenshot:
        DSB18/
        ├── {SampleID_Hash}/
        │   ├── images/
        │   │   └── {SampleID_Hash}.png
        │   └── masks/
        │       ├── mask_0.png
        │       ├── mask_1.png
        │       └── ...
    """
    dataset_dir = DATA_ROOT / "DSB18"
    
    if not dataset_dir.exists():
        print(f"Error: DSB18 dir not found at {dataset_dir}")
        return

    sample_ids = sorted([d for d in os.listdir(dataset_dir) 
                         if (dataset_dir / d).is_dir()])

    if max_samples:
        sample_ids = sample_ids[:max_samples]

    for sample_id in sample_ids:
        sample_path = dataset_dir / sample_id
        image_dir = sample_path / "images"
        mask_dir = sample_path / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            continue

        img_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        if not img_files:
            continue
            
        img_path = image_dir / img_files[0]
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Error loading image for {sample_id}: {e}")
            continue

        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)
        
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        if not mask_files:
            continue

        for mask_file in mask_files:
            m_path = mask_dir / mask_file
            try:
                m = np.array(Image.open(m_path).convert('L'))
                if m.shape != (h, w):
                    m = np.array(Image.open(m_path).resize((w, h), Image.NEAREST).convert('L'))
                full_mask = np.maximum(full_mask, m)
            except Exception as e:
                pass
        
        gt_mask = (full_mask > 127).astype(np.uint8)

        yield Sample(
            image=image,
            gt_mask=gt_mask,
            dataset_name="DSB18",
            sample_id=sample_id,
            text_prompt=DATASET_PROMPTS["DSB18"]
        )

# Dataset loader registry
DATASET_LOADERS = {
    "CHASE_DB1": load_chase_db1,
    "STARE": load_stare,
    "CVC-ClinicDB": load_cvc_clinicdb,
    "ETIS-Larib": load_etis_larib,
    "PH2": load_ph2,
    "TN3K": load_tn3k,
    "TG3K": load_tg3k,
    "DDTI": load_ddti,
    "HC18": load_hc18,
    "DSB18": load_dsb18,
    "CoNSeP": load_consep,
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
