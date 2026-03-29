"""
CLI Main Entry Point

This module provides the command-line interface for the nnUNet framework.
"""

import argparse
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Unified nnUNet Segmentation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Segment single image
  nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t gtrc
  
  # Batch processing
  nnunetsegmentator batch -i input_dir/ -o output_dir/ -t deep_psma --num-workers 4
  
  # List available tasks
  nnunetsegmentator list-tasks
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment single image')
    segment_parser.add_argument('-i', '--input', required=True, 
                               help='Input image path (NIfTI or DICOM directory)')
    segment_parser.add_argument('-o', '--output', required=True,
                               help='Output segmentation path')
    segment_parser.add_argument('-t', '--task', required=True,
                               help='Task name (e.g., gtrc, lion, deep_psma)')
    segment_parser.add_argument('-m', '--model', 
                               help='Model path (overrides task default)')
    segment_parser.add_argument('-c', '--config',
                               help='Configuration file path')
    segment_parser.add_argument('--no-labels', action='store_true',
                               help='Do not save individual label masks')
    segment_parser.add_argument('--compute-metrics', action='store_true',
                               help='Compute segmentation metrics')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch processing')
    batch_parser.add_argument('-i', '--input', required=True,
                             help='Input directory or file list')
    batch_parser.add_argument('-o', '--output', required=True,
                             help='Output directory')
    batch_parser.add_argument('-t', '--task', required=True,
                             help='Task name')
    batch_parser.add_argument('--num-workers', type=int, default=1,
                             help='Number of parallel workers')
    batch_parser.add_argument('--multiprocessing', action='store_true',
                             help='Use multiprocessing instead of threading')
    batch_parser.add_argument('-c', '--config',
                             help='Configuration file path')
    
    # List tasks command
    list_parser = subparsers.add_parser('list-tasks', 
                                       help='List available tasks')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show task information')
    info_parser.add_argument('-t', '--task', required=True,
                             help='Task name')

    # Install local models command
    install_models_parser = subparsers.add_parser(
        'install-models',
        help='Install task models from local archive/directory paths',
    )
    install_models_parser.add_argument(
        '-t',
        '--task',
        required=True,
        help='Task name',
    )
    install_models_parser.add_argument(
        '--model',
        action='append',
        required=True,
        metavar='ID_OR_NAME=PATH',
        help='Model mapping (repeatable). Key can be model name or task ID.',
    )
    install_models_parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing installed model directory',
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Import built-in tasks package to trigger auto-discovery and registration.
    from .. import tasks as _tasks  # noqa: F401
    
    # Execute command
    from ..core import nnunetsegmentatorError

    try:
        if args.command == 'segment':
            cmd_segment(args)
        elif args.command == 'batch':
            cmd_batch(args)
        elif args.command == 'list-tasks':
            cmd_list_tasks(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'install-models':
            cmd_install_models(args)
    except (nnunetsegmentatorError, ValueError, RuntimeError, OSError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)


def cmd_segment(args):
    """Execute segment command"""
    from ..core import SegmentationOrchestrator, Config, TaskRegistry, TaskNotFoundError
    
    # Load configuration
    config = Config.from_file(args.config) if args.config else Config()
    
    # Check if task is registered
    try:
        TaskRegistry.get_task(args.task)
    except TaskNotFoundError:
        logger.error(f"Task '{args.task}' not found. Available tasks: {list(TaskRegistry.list_tasks().keys())}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=args.task,
        config=config,
        model_path=args.model
    )
    
    # Run segmentation
    logger.info(f"Segmenting {args.input} with task {args.task}")
    
    result = orchestrator.segment(
        args.input,
        args.output,
        return_labels=not args.no_labels,
        compute_metrics=args.compute_metrics
    )
    
    logger.info(f"Segmentation saved to: {args.output}")
    
    if args.compute_metrics and result.metrics:
        logger.info("Metrics:")
        for key, value in result.metrics.items():
            logger.info(f"  {key}: {value:.2f}")


def cmd_batch(args):
    """Execute batch command"""
    from ..core import SegmentationOrchestrator, Config
    from pathlib import Path
    
    # Load configuration
    config = Config.from_file(args.config) if args.config else Config()
    
    # Get input list
    input_path = Path(args.input)
    if input_path.is_dir():
        # Find all NIfTI files
        input_list = list(input_path.glob("*.nii.gz")) + list(input_path.glob("*.nii"))
    else:
        # Assume it's a file with list of paths
        with open(input_path) as f:
            input_list = [line.strip() for line in f if line.strip()]
    
    if not input_list:
        logger.error(f"No input files found in {args.input}")
        sys.exit(1)
    
    logger.info(f"Found {len(input_list)} input files")
    
    # Create orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=args.task,
        config=config
    )
    
    # Progress callback
    def progress_callback(current, total, result):
        logger.info(f"Progress: {current}/{total}")
    
    # Run batch processing
    results = orchestrator.segment_batch(
        input_list,
        args.output,
        num_workers=args.num_workers,
        use_multiprocessing=args.multiprocessing,
        progress_callback=progress_callback
    )
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Batch processing complete: {successful}/{len(input_list)} successful")


def cmd_list_tasks(args):
    """Execute list-tasks command"""
    from ..core import TaskRegistry
    
    tasks = TaskRegistry.list_tasks()
    
    if not tasks:
        print("No tasks registered.")
        return
    
    print("Available tasks:")
    for name, description in tasks.items():
        print(f"  {name}: {description}")


def cmd_info(args):
    """Execute info command"""
    from ..core import TaskRegistry
    
    task = TaskRegistry.get_task(args.task)
    
    print(f"Task: {task.name}")
    print(f"\nModels:")
    for model_name, model_info in task.models.items():
        print(f"  {model_name}:")
        print(f"    Task ID: {model_info.task_id}")
        print(f"    Modality: {model_info.modality}")
        print(f"    Description: {model_info.description}")
        print(f"    Labels: {model_info.labels}")
    
    print(f"\nInput requirements:")
    for key, value in task.input_requirements.items():
        print(f"  {key}: {value}")
    
    print(f"\nOutput config:")
    for key, value in task.output_config.items():
        print(f"  {key}: {value}")


def _parse_model_mappings(entries):
    """
    Parse --model entries of the form KEY=PATH.
    """
    mappings = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --model value '{entry}'. Expected format: ID_OR_NAME=PATH"
            )
        key, raw_path = entry.split("=", 1)
        key = key.strip()
        raw_path = raw_path.strip()
        if not key or not raw_path:
            raise ValueError(
                f"Invalid --model value '{entry}'. Expected format: ID_OR_NAME=PATH"
            )
        mappings[key] = raw_path
    return mappings


def cmd_install_models(args):
    """Execute install-models command"""
    from ..core import TaskRegistry

    model_sources = _parse_model_mappings(args.model)
    installed = TaskRegistry.install_task_models_from_local(
        task_name=args.task,
        model_sources=model_sources,
        force=args.force,
    )
    logger.info("Installed %d model(s) for task '%s'", len(installed), args.task)
    for model_name, path in installed.items():
        logger.info("  %s -> %s", model_name, path)


if __name__ == '__main__':
    main()
