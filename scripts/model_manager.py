#!/usr/bin/env python3
"""
Model Management Script for nnunetsegmentator

This script provides utilities for managing model weights, including:
- Listing available models
- Downloading models
- Checking model status
- Organizing model directories

Usage:
    python model_manager.py --list
    python model_manager.py --download --task gtrc
    python model_manager.py --status
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from nnunetsegmentator.core.registry import TaskRegistry
    from nnunetsegmentator.core.config import Config
except ImportError:
    print("Error: nnunetsegmentator not found. Please install it first.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_available_models():
    """List all available models from registered tasks."""
    print("\n" + "=" * 80)
    print("Available Models")
    print("=" * 80)
    
    tasks = TaskRegistry.list_tasks()
    
    for task_name in sorted(tasks.keys()):
        task = TaskRegistry.get_task(task_name)
        print(f"\nTask: {task_name}")
        print(f"Description: {tasks[task_name]}")
        print(f"Models:")
        
        for model_name, model_info in task.models.items():
            print(f"  - {model_name}")
            print(f"    Task ID: {model_info.task_id}")
            print(f"    Modality: {model_info.modality}")
            print(f"    Labels: {len(model_info.labels)} classes")
            print(f"    URL: {model_info.url}")
    
    print("\n" + "=" * 80)


def check_model_status(config: Config) -> Dict[str, Dict]:
    """
    Check status of downloaded models.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with model status information
    """
    model_dir = config.model_dir
    status = {}
    
    tasks = TaskRegistry.list_tasks()
    
    for task_name in tasks.keys():
        task = TaskRegistry.get_task(task_name)
        task_status = {
            'downloaded': False,
            'models': {},
            'path': str(model_dir / task_name)
        }
        
        # Check if task directory exists
        task_path = model_dir / task_name
        if task_path.exists():
            task_status['downloaded'] = True
            
            # Check each model
            for model_name, model_info in task.models.items():
                model_path = task_path / model_info.task_id
                task_status['models'][model_name] = {
                    'exists': model_path.exists(),
                    'path': str(model_path)
                }
        
        status[task_name] = task_status
    
    return status


def print_model_status(config: Config):
    """Print model status in a formatted table."""
    status = check_model_status(config)
    
    print("\n" + "=" * 80)
    print("Model Status")
    print("=" * 80)
    print(f"Model Directory: {config.model_dir}")
    print()
    
    for task_name, task_status in sorted(status.items()):
        status_icon = "✓" if task_status['downloaded'] else "✗"
        print(f"{status_icon} {task_name:20s} - {task_status['path']}")
        
        if task_status['downloaded']:
            for model_name, model_info in task_status['models'].items():
                model_icon = "✓" if model_info['exists'] else "✗"
                print(f"  {model_icon} {model_name}")
    
    print("\n" + "=" * 80)


def download_model(task_name: str, config: Config, variant: str = None):
    """
    Download model for a specific task.
    
    Args:
        task_name: Task name
        config: Configuration object
        variant: Model variant (if applicable)
    """
    import subprocess
    
    # Get download script path
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_models.py"
    
    if not download_script.exists():
        logger.error(f"Download script not found: {download_script}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(download_script),
        "--task", task_name,
        "--output-dir", str(config.model_dir)
    ]
    
    if variant:
        cmd.extend(["--variant", variant])
    
    logger.info(f"Downloading model for task: {task_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"Model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False


def clean_model_cache(config: Config, dry_run: bool = True):
    """
    Clean model cache directory.
    
    Args:
        config: Configuration object
        dry_run: If True, only show what would be deleted
    """
    cache_dir = config.cache_dir
    
    if not cache_dir.exists():
        logger.info("Cache directory does not exist")
        return
    
    # Find all files in cache
    cache_files = list(cache_dir.rglob('*'))
    cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
    
    print(f"\nCache Directory: {cache_dir}")
    print(f"Total Files: {len([f for f in cache_files if f.is_file()])}")
    print(f"Total Size: {cache_size / (1024**3):.2f} GB")
    
    if dry_run:
        print("\nDry run - no files will be deleted")
        print("Run with --force to actually delete files")
    else:
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("\nCache cleaned successfully")


def install_local_models(
    task_name: str,
    model_pairs: List[str],
    force: bool = False,
):
    """
    Install model(s) for a task from local archive/directory paths.
    """
    if not model_pairs:
        raise ValueError("--model is required for --install-local")

    model_sources = {}
    for pair in model_pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid --model value '{pair}'. Expected: DATASETID_MODELNAME=PATH"
            )
        key, raw_path = pair.split("=", 1)
        key = key.strip()
        raw_path = raw_path.strip()
        if not key or not raw_path:
            raise ValueError(
                f"Invalid --model value '{pair}'. Expected: DATASETID_MODELNAME=PATH"
            )
        model_sources[key] = raw_path

    installed = TaskRegistry.install_task_models_from_local(
        task_name=task_name,
        model_sources=model_sources,
        force=force,
    )
    logger.info("Installed %d local model(s) for task '%s'", len(installed), task_name)
    for model_name, path in installed.items():
        logger.info("  %s -> %s", model_name, path)


def migrate_workspace_models(
    config: Config,
    workspace_results: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Migrate models from nnUNet workspace results into canonical model storage.
    """
    import subprocess

    script_dir = Path(__file__).parent
    migrate_script = script_dir / "migrate_workspace_models.py"

    if not migrate_script.exists():
        logger.error("Migration script not found: %s", migrate_script)
        return False

    cmd = [
        sys.executable,
        str(migrate_script),
        "--model-root",
        str(config.model_dir),
    ]
    if workspace_results:
        cmd.extend(["--workspace-results", workspace_results])
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Migrating workspace models into canonical model root: %s", config.model_dir)
    logger.info("Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Workspace model migration failed: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Model management for nnunetsegmentator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python model_manager.py --list
  
  # Check model download status
  python model_manager.py --status
  
  # Download a specific model
  python model_manager.py --download --task gtrc
  
  # Download with variant
  python model_manager.py --download --task lion --variant psma

  # Install local model files (canonical key format)
  python model_manager.py --install-local --task total --model Dataset291_total_organs=/path/to/total_organs.zip

  # Preview workspace-results migration into canonical model root
  python model_manager.py --migrate-workspace --dry-run

  # Run workspace-results migration and overwrite existing targets
  python model_manager.py --migrate-workspace --force

  # Clean cache (dry run)
  python model_manager.py --clean-cache
  
  # Clean cache (force)
  python model_manager.py --clean-cache --force
        """
    )
    
    # Action options
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--list',
        action='store_true',
        help='List all available models'
    )
    action_group.add_argument(
        '--status',
        action='store_true',
        help='Check model download status'
    )
    action_group.add_argument(
        '--download',
        action='store_true',
        help='Download model for a task'
    )
    action_group.add_argument(
        '--clean-cache',
        action='store_true',
        help='Clean model cache directory'
    )
    action_group.add_argument(
        '--install-local',
        action='store_true',
        help='Install task model(s) from local archive/directory paths'
    )
    action_group.add_argument(
        '--migrate-workspace',
        action='store_true',
        help='Migrate models from nnUNet workspace results into canonical model layout'
    )
    
    # Task options
    parser.add_argument(
        '--task',
        type=str,
        help='Task name for download'
    )
    parser.add_argument(
        '--variant',
        type=str,
        help='Model variant (e.g., psma, fdg for LION)'
    )
    parser.add_argument(
        '--model',
        action='append',
        metavar='DATASETID_MODELNAME=PATH',
        help='Model mapping for --install-local (repeatable)'
    )
    parser.add_argument(
        '--workspace-results',
        type=str,
        help='Path to nnUNet workspace results directory for --migrate-workspace'
    )
    
    # Other options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force action (overwrite existing targets or delete cache)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview action without writing changes (supported by --migrate-workspace)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        help='Override default model directory'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    if args.model_dir:
        config.model_dir = Path(args.model_dir)
    
    # Execute action
    if args.list:
        list_available_models()
    
    elif args.status:
        print_model_status(config)
    
    elif args.download:
        if not args.task:
            parser.error("--task is required for download")
        download_model(args.task, config, args.variant)
    
    elif args.clean_cache:
        clean_model_cache(config, dry_run=not args.force)

    elif args.install_local:
        if not args.task:
            parser.error("--task is required for --install-local")
        install_local_models(args.task, args.model or [], force=args.force)

    elif args.migrate_workspace:
        migrate_workspace_models(
            config=config,
            workspace_results=args.workspace_results,
            force=args.force,
            dry_run=args.dry_run,
        )


if __name__ == '__main__':
    main()
