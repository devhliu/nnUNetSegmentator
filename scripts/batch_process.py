#!/usr/bin/env python3
"""
Batch Processing Script for nnunetsegmentator

This script provides a convenient command-line interface for batch processing
of medical images using nnunetsegmentator.

Usage:
    python batch_process.py --input-dir /path/to/images --output-dir /path/to/output --task total
    python batch_process.py --input-list file_list.txt --output-dir /path/to/output --task lion
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from nnunetsegmentator import SegmentationOrchestrator, Config
    from nnunetsegmentator.core.registry import TaskRegistry
except ImportError:
    print("Error: nnunetsegmentator not found. Please install it first.")
    print("Run: pip install -e .")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_supported_extensions():
    """Get list of supported image file extensions."""
    return ['.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.dcm']


def find_image_files(input_dir: Path) -> List[Path]:
    """
    Find all image files in a directory.
    
    Args:
        input_dir: Input directory path
        
    Returns:
        List of image file paths
    """
    image_files = []
    extensions = get_supported_extensions()
    
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'**/*{ext}'))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    logger.info(f"Found {len(image_files)} image files in {input_dir}")
    return image_files


def load_file_list(file_list_path: Path) -> List[Path]:
    """
    Load list of files from a text file.
    
    Args:
        file_list_path: Path to file list
        
    Returns:
        List of file paths
    """
    files = []
    with open(file_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                files.append(Path(line))
    
    logger.info(f"Loaded {len(files)} files from {file_list_path}")
    return files


def process_batch(
    input_files: List[Path],
    output_dir: Path,
    task_name: str,
    num_workers: int = 1,
    config: Config = None,
    save_metadata: bool = True
) -> Dict[str, str]:
    """
    Process a batch of images.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        task_name: Task name
        num_workers: Number of parallel workers
        config: Configuration object
        save_metadata: Whether to save metadata
        
    Returns:
        Dictionary mapping input files to output files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=task_name,
        config=config
    )
    
    # Process batch
    results = orchestrator.segment_batch(
        input_list=[str(f) for f in input_files],
        output_dir=str(output_dir),
        num_workers=num_workers,
        use_multiprocessing=(num_workers > 1)
    )
    
    # Create results mapping
    results_map = {}
    for i, result in enumerate(results):
        input_file = input_files[i]
        output_file = output_dir / f"{input_file.stem}_seg.nii.gz"
        results_map[str(input_file)] = str(output_file)
        
        # Save metadata if requested
        if save_metadata and result.metadata:
            metadata_file = output_dir / f"{input_file.stem}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(result.metadata, f, indent=2)
    
    return results_map


def main():
    parser = argparse.ArgumentParser(
        description='Batch processing with nnunetsegmentator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory
  python batch_process.py --input-dir /data/images --output-dir /data/output --task total
  
  # Process specific files from a list
  python batch_process.py --input-list files.txt --output-dir /data/output --task lion
  
  # Use multiple workers
  python batch_process.py --input-dir /data/images --output-dir /data/output --task total --workers 4
  
  # List available tasks
  python batch_process.py --list-tasks
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing images'
    )
    input_group.add_argument(
        '--input-list',
        type=str,
        help='Text file containing list of input files'
    )
    input_group.add_argument(
        '--list-tasks',
        action='store_true',
        help='List available tasks and exit'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for segmentations'
    )
    
    # Task options
    parser.add_argument(
        '--task',
        type=str,
        help='Task name (e.g., total, lion, gtrc, deep_psma)'
    )
    
    # Processing options
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Do not save metadata files'
    )
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nAvailable Tasks:")
        print("=" * 60)
        tasks = TaskRegistry.list_tasks()
        for name, description in tasks.items():
            print(f"  {name:20s} - {description}")
        print("=" * 60)
        return
    
    # Validate arguments
    if not args.task:
        parser.error("--task is required when processing images")
    if not args.output_dir:
        parser.error("--output-dir is required when processing images")
    
    # Get input files
    if args.input_dir:
        input_files = find_image_files(Path(args.input_dir))
    else:
        input_files = load_file_list(Path(args.input_list))
    
    if not input_files:
        logger.error("No input files found")
        sys.exit(1)
    
    # Create configuration
    config = Config(
        use_gpu=True,
        gpu_id=args.gpu,
        num_workers=args.workers
    )
    
    # Process batch
    logger.info(f"Processing {len(input_files)} files with task '{args.task}'")
    logger.info(f"Using {args.workers} worker(s)")
    
    try:
        results = process_batch(
            input_files=input_files,
            output_dir=Path(args.output_dir),
            task_name=args.task,
            num_workers=args.workers,
            config=config,
            save_metadata=not args.no_metadata
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Processed: {len(results)} files")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
