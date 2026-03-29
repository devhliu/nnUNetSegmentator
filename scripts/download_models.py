#!/usr/bin/env python3
"""
Model Weight Download Script

This script downloads pretrained model weights for nnunetsegmentator tasks.

Model Sources:
- GTRC-Net: GitHub (Git LFS) - https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained
- LION: GitHub Releases - https://github.com/ENHANCE-PET/LION/releases
- DEEP-PSMA: GitHub (Git LFS) - https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA

Usage:
    python download_models.py --task gtrc --output-dir ./models
    python download_models.py --task all --output-dir ./models
    python download_models.py --list
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Please install required packages: pip install requests tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DownloadMethod(Enum):
    """Download method type."""
    DIRECT = "direct"       # Direct HTTP download
    GIT_LFS = "git_lfs"     # Git LFS clone
    PIP = "pip"             # pip install


@dataclass
class ModelInfo:
    """Model download information."""
    name: str
    task_id: str
    url: str
    checksum: Optional[str]
    filename: str
    description: str
    method: DownloadMethod = DownloadMethod.DIRECT
    github_repo: Optional[str] = None
    target_subdir: Optional[str] = None  # Subdirectory within repo to extract
    variants: Optional[Dict[str, str]] = None  # Model variants (e.g., PSMA, FDG)


# Available models with actual download sources
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    'gtrc': ModelInfo(
        name='gtrc',
        task_id='Dataset881_PSMA_PET',
        url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git',
        checksum=None,  # Git LFS handles integrity
        filename='gtrc.zip',
        description='GTRC-Net: Glioblastoma treatment response segmentation (PET/CT)',
        method=DownloadMethod.GIT_LFS,
        github_repo='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git',
        target_subdir='data/nnUNet_data/results',
        variants={
            'psma': 'Dataset881_PSMA_PET',
            'fdg': 'Dataset882_FDG_PET',
            'lupsma': 'Dataset883_LUPSMA_SPECT',
        }
    ),
    'lion': ModelInfo(
        name='lion',
        task_id='Dataset789_Tumors',
        url='https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_fdg_5235_17122025.zip',
        checksum=None,
        filename='lion_fdg.zip',
        description='LION: PET lesion segmentation (FDG PET, 5,235 patients)',
        method=DownloadMethod.DIRECT,
        variants={
            'fdg': 'https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_fdg_5235_17122025.zip',
            'psma': 'https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_psma_2046_25112025.zip',
        }
    ),
    'deep_psma': ModelInfo(
        name='deep_psma',
        task_id='Dataset881_PSMA_PET',
        url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA.git',
        checksum=None,
        filename='deep_psma.zip',
        description='DEEP-PSMA: PSMA PET lesion segmentation (trained on DEEP-PSMA Challenge)',
        method=DownloadMethod.GIT_LFS,
        github_repo='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA.git',
        target_subdir='data/nnUNet_data/results',
    ),
    'total': ModelInfo(
        name='total',
        task_id='Dataset291',
        url='https://github.com/wasserth/TotalSegmentator',
        checksum=None,
        filename='total.zip',
        description='TotalSegmentator: Whole-body CT segmentation (117 structures)',
        method=DownloadMethod.PIP,
        variants={
            'total': 'Default 117-class CT segmentation',
            'total_mr': '50-class MR segmentation',
            'total_fast': 'Fast 3mm resolution',
            'total_fastest': 'Fastest 6mm resolution',
            'lung_vessels': 'Lung vessel segmentation',
            'body': 'Body region segmentation',
            'vertebrae_mr': 'Vertebrae MR segmentation',
        }
    ),
    'total_mr': ModelInfo(
        name='total_mr',
        task_id='Dataset850',
        url='https://github.com/wasserth/TotalSegmentator',
        checksum=None,
        filename='total_mr.zip',
        description='TotalSegmentator MR: Whole-body MR segmentation (50 structures)',
        method=DownloadMethod.PIP,
    ),
    'ts2d': ModelInfo(
        name='ts2d',
        task_id='Dataset900',
        url='https://zenodo.org/records/16985939',
        checksum=None,
        filename='ts2d_v2.zip',
        description='TotalSegmentator 2D: Fast 2D projection-based segmentation (117 structures)',
        method=DownloadMethod.DIRECT,
        variants={
            'v2': 'https://zenodo.org/records/16985939',
            'v1': 'https://zenodo.org/records/16574232',
        }
    ),
    'dukeseg': ModelInfo(
        name='dukeseg',
        task_id='Dataset1004',
        url='https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git',
        checksum=None,
        filename='dukeseg.zip',
        description='DukeSeg: Comprehensive segmentation (140 structures)',
        method=DownloadMethod.GIT_LFS,
        github_repo='https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git',
        variants={
            'skeleton': 'Dataset1004 - 62 bone structures',
            'model2': 'Dataset1001 - 61 head/neck/thorax/muscles',
            'model3': 'Dataset1002 - 17 abdominal organs',
        }
    ),
}


def compute_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute checksum of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (sha256, md5)

    Returns:
        Hexadecimal checksum string
    """
    if algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'md5':
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def check_git_lfs_installed() -> bool:
    """Check if Git LFS is installed."""
    try:
        result = subprocess.run(
            ['git', 'lfs', 'version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_file(
    url: str,
    output_path: Path,
    expected_checksum: Optional[str] = None
) -> bool:
    """
    Download a file with progress bar.

    Args:
        url: Download URL
        output_path: Output file path
        expected_checksum: Expected file checksum for verification

    Returns:
        True if download successful
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from: {url}")
    logger.info(f"Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify checksum if provided
        if expected_checksum:
            logger.info("Verifying checksum...")
            actual_checksum = compute_checksum(output_path)
            if actual_checksum != expected_checksum:
                logger.error(
                    f"Checksum mismatch!\n"
                    f"Expected: {expected_checksum}\n"
                    f"Got: {actual_checksum}"
                )
                output_path.unlink()
                return False
            logger.info("Checksum verified successfully")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def clone_git_lfs_repo(
    repo_url: str,
    output_dir: Path,
    target_subdir: Optional[str] = None
) -> Optional[Path]:
    """
    Clone a Git LFS repository.

    Args:
        repo_url: Git repository URL
        output_dir: Output directory
        target_subdir: Subdirectory to extract from repo

    Returns:
        Path to model directory if successful
    """
    if not check_git_lfs_installed():
        logger.error("Git LFS is not installed. Please install it first:")
        logger.error("  - Ubuntu/Debian: sudo apt-get install git-lfs")
        logger.error("  - macOS: brew install git-lfs")
        logger.error("  - Windows: https://git-lfs.github.com/")
        return None

    repo_name = repo_url.split('/')[-1].replace('.git', '')
    clone_dir = output_dir / repo_name

    logger.info(f"Cloning Git LFS repository: {repo_url}")
    logger.info(f"Target directory: {clone_dir}")

    try:
        # Clone with Git LFS
        result = subprocess.run(
            ['git', 'lfs', 'clone', repo_url, str(clone_dir)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large models
        )

        if result.returncode != 0:
            logger.error(f"Git LFS clone failed: {result.stderr}")
            return None

        logger.info("Git LFS clone completed successfully")

        # If target_subdir specified, copy that directory
        if target_subdir:
            source_path = clone_dir / target_subdir
            if source_path.exists():
                # Copy to output_dir
                for item in source_path.iterdir():
                    dest = output_dir / item.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                logger.info(f"Model files copied from: {source_path}")
                return output_dir
            else:
                logger.warning(f"Target subdirectory not found: {source_path}")
                return clone_dir

        return clone_dir

    except subprocess.TimeoutExpired:
        logger.error("Git LFS clone timed out")
        return None
    except Exception as e:
        logger.error(f"Git LFS clone failed: {e}")
        return None


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """
    Extract a zip archive.

    Args:
        archive_path: Path to zip file
        extract_dir: Directory to extract to

    Returns:
        True if extraction successful
    """
    logger.info(f"Extracting {archive_path} to {extract_dir}")

    try:
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)

        logger.info("Extraction complete")
        return True

    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        return False
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def organize_nnunet_model(extract_dir: Path, output_dir: Path, task_id: str) -> Path:
    """
    Organize extracted nnUNet model into correct structure.

    nnUNet expects models in:
    nnUNet_results/DatasetXXX_NAME/nnUNetTrainer__xxx/fold_X/

    Args:
        extract_dir: Extracted archive directory
        output_dir: Output directory
        task_id: Task ID (e.g., Dataset789_Tumors)

    Returns:
        Path to organized model directory
    """
    model_dir = output_dir / task_id

    # Find the actual model directory in the extracted content
    possible_paths = [
        extract_dir / task_id,
        extract_dir / 'nnUNet_results' / task_id,
    ]

    for path in possible_paths:
        if path.exists():
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.copytree(path, model_dir)
            logger.info(f"Model organized to: {model_dir}")
            return model_dir

    # If not found, check for any Dataset directory
    for item in extract_dir.rglob('Dataset*'):
        if item.is_dir():
            target = output_dir / item.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
            logger.info(f"Model organized to: {target}")
            return target

    # Return extract_dir if nothing found
    logger.warning(f"Could not find expected model structure, using: {extract_dir}")
    return extract_dir


def download_model(
    task_name: str,
    output_dir: Path,
    variant: Optional[str] = None,
    extract: bool = True,
    keep_archive: bool = False
) -> Optional[Path]:
    """
    Download model weights for a specific task.

    Args:
        task_name: Task name (gtrc, lion, deep_psma)
        output_dir: Output directory for model weights
        variant: Model variant (e.g., 'psma', 'fdg' for LION)
        extract: Whether to extract the archive
        keep_archive: Whether to keep the archive after extraction

    Returns:
        Path to model directory if successful, None otherwise
    """
    if task_name not in AVAILABLE_MODELS:
        logger.error(f"Unknown task: {task_name}")
        logger.info(f"Available tasks: {list(AVAILABLE_MODELS.keys())}")
        return None

    model = AVAILABLE_MODELS[task_name]

    logger.info(f"Downloading model: {model.name}")
    logger.info(f"Task ID: {model.task_id}")
    logger.info(f"Description: {model.description}")

    # Handle variants
    download_url = model.url
    if variant and model.variants and variant in model.variants:
        download_url = model.variants[variant]
        logger.info(f"Using variant: {variant}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if model.method == DownloadMethod.GIT_LFS:
        # Git LFS clone
        model_path = clone_git_lfs_repo(
            model.github_repo,
            output_dir,
            model.target_subdir
        )
        return model_path
    
    elif model.method == DownloadMethod.PIP:
        # PIP install (for TotalSegmentator)
        logger.info(f"Installing via pip: {model.name}")
        logger.info("Note: Models will be downloaded automatically on first use")
        
        try:
            # Check if TotalSegmentator is already installed
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', 'TotalSegmentator'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Install TotalSegmentator
                logger.info("Installing TotalSegmentator...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', 'TotalSegmentator'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to install TotalSegmentator: {result.stderr}")
                    return None
                
                logger.info("TotalSegmentator installed successfully")
            else:
                logger.info("TotalSegmentator already installed")
            
            # Create a marker file to indicate models are available
            marker_file = output_dir / f"{model.name}_installed.txt"
            marker_file.write_text(f"TotalSegmentator installed for {model.name}\n")
            logger.info(f"Models will be downloaded to: ~/.totalsegmentator/nnunet/results/")
            
            return output_dir
            
        except Exception as e:
            logger.error(f"PIP installation failed: {e}")
            return None

    elif model.method == DownloadMethod.DIRECT:
        # Direct HTTP download
        archive_path = output_dir / model.filename

        if archive_path.exists():
            logger.info(f"Archive already exists: {archive_path}")
            if model.checksum:
                actual_checksum = compute_checksum(archive_path)
                if actual_checksum != model.checksum:
                    logger.warning("Existing archive checksum mismatch, re-downloading")
                    archive_path.unlink()
                    if not download_file(download_url, archive_path, model.checksum):
                        return None
        else:
            if not download_file(download_url, archive_path, model.checksum):
                return None

        # Extract if requested
        if extract:
            extract_dir = output_dir / f"{model.filename}_extracted"
            if not extract_archive(archive_path, extract_dir):
                return None

            # Organize into nnUNet structure
            model_path = organize_nnunet_model(extract_dir, output_dir, model.task_id)

            # Cleanup
            shutil.rmtree(extract_dir)
            if not keep_archive:
                archive_path.unlink()
                logger.info(f"Removed archive: {archive_path}")

            return model_path

        return archive_path

    return None


def download_all_models(
    output_dir: Path,
    extract: bool = True,
    keep_archive: bool = False
) -> Dict[str, Path]:
    """
    Download all available models.

    Args:
        output_dir: Output directory
        extract: Whether to extract archives
        keep_archive: Whether to keep archives

    Returns:
        Dictionary mapping task names to model paths
    """
    results = {}

    for task_name in AVAILABLE_MODELS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {task_name}")
        logger.info('='*60)

        model_path = download_model(
            task_name,
            output_dir,
            extract=extract,
            keep_archive=keep_archive
        )

        if model_path:
            results[task_name] = model_path
        else:
            logger.error(f"Failed to download: {task_name}")

    return results


def list_available_models() -> None:
    """Print list of available models."""
    print("\n" + "=" * 70)
    print("Available Models")
    print("=" * 70)

    for task_name, model in AVAILABLE_MODELS.items():
        print(f"\n  {task_name}:")
        print(f"    Name:        {model.name}")
        print(f"    Task ID:     {model.task_id}")
        print(f"    Description: {model.description}")
        print(f"    Method:      {model.method.value}")
        print(f"    URL:         {model.url}")
        if model.variants:
            print(f"    Variants:    {', '.join(model.variants.keys())}")

    print("\n" + "=" * 70)
    print("\nNotes:")
    print("  - GTRC-Net and DEEP-PSMA require Git LFS to be installed")
    print("  - LION models are downloaded from GitHub releases")
    print("  - Use --variant option to download specific model variants")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download pretrained model weights for nnunetsegmentator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python download_models.py --list

  # Download GTRC-Net (requires Git LFS)
  python download_models.py --task gtrc --output-dir ./models

  # Download LION FDG model
  python download_models.py --task lion --output-dir ./models

  # Download LION PSMA variant
  python download_models.py --task lion --output-dir ./models --variant psma

  # Download all models
  python download_models.py --task all --output-dir ./models

  # Download without extracting
  python download_models.py --task lion --output-dir ./models --no-extract

Model Sources:
  - GTRC-Net: https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained
  - LION: https://github.com/ENHANCE-PET/LION/releases
  - DEEP-PSMA: https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA
        """
    )

    parser.add_argument(
        '--task',
        type=str,
        choices=list(AVAILABLE_MODELS.keys()) + ['all'],
        help='Task name to download (or "all" for all models)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Output directory for model weights (default: ./models)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        help='Model variant (e.g., psma, fdg for LION)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Do not extract the archive after download'
    )
    parser.add_argument(
        '--keep-archive',
        action='store_true',
        help='Keep the archive after extraction'
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if not args.task:
        parser.print_help()
        return

    output_dir = Path(args.output_dir)

    if args.task == 'all':
        results = download_all_models(
            output_dir,
            extract=not args.no_extract,
            keep_archive=args.keep_archive
        )

        print("\n" + "=" * 60)
        print("Download Summary:")
        print("=" * 60)
        for task_name, path in results.items():
            print(f"  {task_name}: {path}")
        print(f"\nSuccessfully downloaded: {len(results)}/{len(AVAILABLE_MODELS)}")

    else:
        model_path = download_model(
            args.task,
            output_dir,
            variant=args.variant,
            extract=not args.no_extract,
            keep_archive=args.keep_archive
        )

        if model_path:
            print(f"\nModel downloaded successfully: {model_path}")
        else:
            print("\nDownload failed!")
            sys.exit(1)


if __name__ == '__main__':
    main()
