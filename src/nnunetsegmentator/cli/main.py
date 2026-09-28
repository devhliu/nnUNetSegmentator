"""
CLI Main Entry Point

This module provides the command-line interface for the nnUNet framework.
"""

import argparse
import sys
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

  # Segment with automatic model selection based on target organs
  nnunetsegmentator segment -i input.nii.gz -o output.nii.gz --organs "liver,left kidney" --modality CT

  # Segment with an explicit model subset of a task
  nnunetsegmentator segment -i input.nii.gz -o output.nii.gz -t moose --models clin_ct_organs,clin_ct_ribs

  # Batch processing
  nnunetsegmentator batch -i input_dir/ -o output_dir/ -t deep_psma --num-workers 4

  # List available tasks / organs, find models for organs
  nnunetsegmentator list-tasks
  nnunetsegmentator list-organs -t moose
  nnunetsegmentator find-models --organs "liver,SCT:10200004" --modality CT

  # Download model weights explicitly (inference never downloads implicitly)
  nnunetsegmentator download-models -t moose --models clin_ct_organs

  # Install local weights by explicit canonical key
  nnunetsegmentator install-models -t total --model Dataset291_total_organs=/path/total_organs.zip

  # Auto-discover and install weights from a root directory
  nnunetsegmentator install-models --from /data/weights --dry-run
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment single image')
    segment_parser.add_argument('-i', '--input', required=True,
                               help='Input image path (NIfTI or DICOM directory)')
    segment_parser.add_argument('-o', '--output', required=True,
                               help='Output segmentation path')
    segment_parser.add_argument('-t', '--task',
                               help='Task name (e.g., gtrc, lion, moose); optional when --organs is used')
    segment_parser.add_argument('--organs',
                               help='Comma-separated target organs for automatic model '
                                    'selection (mutually exclusive with -t/--task). '
                                    'Accepts standard names, aliases or SNOMED codes.')
    segment_parser.add_argument('--modality',
                               help='Modality filter for --organs selection (CT/PT/MR)')
    segment_parser.add_argument('--strategy', default='min_models',
                               help='Model selection strategy (default: min_models)')
    segment_parser.add_argument('--models',
                               help='Comma-separated subset of task models to run '
                                    '(e.g. "clin_ct_organs,clin_ct_ribs")')
    segment_parser.add_argument('-m', '--model',
                               help='Model path (overrides task default)')
    segment_parser.add_argument('-c', '--config',
                               help='Configuration file path')
    segment_parser.add_argument('--no-labels', action='store_true',
                               help='Do not save individual label masks')
    segment_parser.add_argument('--compute-metrics', action='store_true',
                               help='Compute segmentation metrics')
    segment_parser.add_argument('--no-convert-nifti', action='store_true',
                               help='Do not save the converted NIfTI when the input '
                                    'is a DICOM series')
    segment_parser.add_argument('--no-dicom-seg', action='store_true',
                               help='Do not export a DICOM-SEG object when the input '
                                    'is a DICOM series')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch processing')
    batch_parser.add_argument('--no-convert-nifti', action='store_true',
                              help='Do not save the converted NIfTI when the input '
                                   'is a DICOM series')
    batch_parser.add_argument('--no-dicom-seg', action='store_true',
                              help='Do not export a DICOM-SEG object when the input '
                                   'is a DICOM series')
    batch_parser.add_argument('-i', '--input', required=True,
                             help='Input directory or file list')
    batch_parser.add_argument('-o', '--output', required=True,
                             help='Output directory')
    batch_parser.add_argument('-t', '--task',
                             help='Task name; optional when --organs is used')
    batch_parser.add_argument('--organs',
                             help='Comma-separated target organs for automatic model '
                                  'selection (mutually exclusive with -t/--task)')
    batch_parser.add_argument('--modality',
                             help='Modality filter for --organs selection (CT/PT/MR)')
    batch_parser.add_argument('--strategy', default='min_models',
                             help='Model selection strategy (default: min_models)')
    batch_parser.add_argument('--models',
                             help='Comma-separated subset of task models to run')
    batch_parser.add_argument('--num-workers', type=int, default=1,
                             help='Number of parallel workers')
    batch_parser.add_argument('--multiprocessing', action='store_true',
                             help='Use multiprocessing instead of threading')
    batch_parser.add_argument('-c', '--config',
                             help='Configuration file path')

    # List tasks command
    subparsers.add_parser('list-tasks', help='List available tasks')

    # List organs command
    list_organs_parser = subparsers.add_parser('list-organs',
                                               help='List canonical organs available across tasks')
    list_organs_parser.add_argument('-t', '--task',
                                    help='Restrict listing to one task')
    list_organs_parser.add_argument('--modality',
                                    help='Filter organs by modality (CT/PT/MR)')

    # Find models command
    find_models_parser = subparsers.add_parser('find-models',
                                               help='Find model(s) covering a set of organs')
    find_models_parser.add_argument('--organs', required=True,
                                    help='Comma-separated organ list (standard names, '
                                         'aliases or SNOMED codes, e.g. "liver,SCT:10200004")')
    find_models_parser.add_argument('--modality',
                                    help='Modality filter (CT/PT/MR)')
    find_models_parser.add_argument('--strategy', default='min_models',
                                    help='Selection strategy (default: min_models)')
    find_models_parser.add_argument('-t', '--task',
                                    help='Restrict selection to one task')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show task information')
    info_parser.add_argument('-t', '--task', required=True,
                             help='Task name')

    # Download models command
    download_models_parser = subparsers.add_parser(
        'download-models',
        help='Download task models from their registered URLs',
    )
    download_models_parser.add_argument(
        '-t',
        '--task',
        required=True,
        help='Task name',
    )
    download_models_parser.add_argument(
        '--models',
        help='Comma-separated model names to download (default: all models of the task)',
    )
    download_models_parser.add_argument(
        '--force',
        action='store_true',
        help='Remove existing installations and re-download',
    )

    # Install local models command
    install_models_parser = subparsers.add_parser(
        'install-models',
        help='Install task models from local paths or auto-discover them under a root',
    )
    install_models_parser.add_argument(
        '-t',
        '--task',
        help='Task name (required with --model; optional filter with --from)',
    )
    install_models_parser.add_argument(
        '--model',
        action='append',
        metavar='DATASETID_MODELNAME=PATH',
        help="Model mapping (repeatable). Key must be canonical format: Dataset<数字>_<model_name>.",
    )
    install_models_parser.add_argument(
        '--from',
        dest='from_root',
        metavar='ROOT',
        help='Auto-discover payload directories/archives under ROOT and install them',
    )
    install_models_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='With --from, only list what would be installed without writing',
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
        elif args.command == 'list-organs':
            cmd_list_organs(args)
        elif args.command == 'find-models':
            cmd_find_models(args)
        elif args.command == 'info':
            cmd_info(args)
        elif args.command == 'download-models':
            cmd_download_models(args)
        elif args.command == 'install-models':
            cmd_install_models(args)
    except (nnunetsegmentatorError, ValueError, RuntimeError, OSError) as exc:
        logger.error("Error: %s", exc)
        sys.exit(1)


def _resolve_task_and_models(args):
    """
    Resolve the task name and model subset from CLI arguments.

    Supports two modes:
        - explicit task:  -t/--task (optionally narrowed by --models)
        - organ-driven:   --organs (task and models selected automatically)

    Returns:
        Tuple of (task_name, models)
    """
    from ..core import TaskRegistry
    from ..mapping import ModelSelector

    models = [m.strip() for m in args.models.split(',')] if getattr(args, 'models', None) else None

    task_name = getattr(args, 'task', None)
    organs = getattr(args, 'organs', None)

    if organs:
        if task_name:
            raise ValueError("--organs and -t/--task are mutually exclusive")
        if models:
            raise ValueError("--organs and --models are mutually exclusive")
        organ_list = [o.strip() for o in organs.split(',') if o.strip()]
        if not organ_list:
            raise ValueError("--organs must contain at least one organ name")

        selector = ModelSelector()
        selector.build_index()
        plan = selector.select(
            organ_list,
            modality=getattr(args, 'modality', None),
            strategy=getattr(args, 'strategy', 'min_models'),
        )
        if not plan.models:
            raise ValueError(
                f"No registered model covers organs {organ_list}. "
                f"Unmatched: {', '.join(plan.unmatched_organs) or '<all>'}"
            )
        task_name = plan.task_name
        models = plan.models
        logger.info(
            "Auto-selected task '%s' models %s for organs %s",
            task_name, models, organ_list,
        )
        if plan.unmatched_organs:
            logger.warning("Unmatched organs (no model provides them): %s",
                           ', '.join(plan.unmatched_organs))
    elif not task_name:
        raise ValueError("Either -t/--task or --organs is required")

    # Validate task exists (raises TaskNotFoundError with available tasks)
    TaskRegistry.get_task(task_name)

    return task_name, models


def cmd_segment(args):
    """Execute segment command"""
    from ..core import SegmentationOrchestrator, Config

    # Load configuration
    config = Config.from_file(args.config) if args.config else Config()

    task_name, models = _resolve_task_and_models(args)

    # Create orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=task_name,
        config=config,
        model_path=args.model,
        models=models
    )

    # Run segmentation
    logger.info(f"Segmenting {args.input} with task {task_name}")

    result = orchestrator.segment(
        args.input,
        args.output,
        return_labels=not args.no_labels,
        compute_metrics=args.compute_metrics,
        save_converted_nifti=not args.no_convert_nifti,
        export_dicom_seg=not args.no_dicom_seg,
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

    task_name, models = _resolve_task_and_models(args)

    # Create orchestrator
    orchestrator = SegmentationOrchestrator(
        task_name=task_name,
        config=config,
        models=models
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
        progress_callback=progress_callback,
        save_converted_nifti=not args.no_convert_nifti,
        export_dicom_seg=not args.no_dicom_seg,
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


def cmd_list_organs(args):
    """Execute list-organs command: canonical organs across registered tasks"""
    from ..mapping import ModelSelector

    selector = ModelSelector()
    index = selector.build_index()

    if not index:
        print("No organs indexed (no task provides labels).")
        return

    print(f"Available organs ({len(index)} canonical names):")
    print(f"{'organ':<38} {'SNOMED':<12} models")
    for organ in sorted(index.keys()):
        providers = index[organ]
        if args.task and not any(p.task_name == args.task for p in providers):
            continue
        if args.modality and not any(
            p.modality.upper() == args.modality.upper() for p in providers
        ):
            continue
        snomed = next((p.snomed_code for p in providers if p.snomed_code), "-")
        model_refs = sorted({f"{p.task_name}/{p.model_name}" for p in providers})
        print(f"{organ:<38} {snomed:<12} {', '.join(model_refs)}")


def cmd_find_models(args):
    """Execute find-models command: select model(s) covering requested organs"""
    import json
    from ..mapping import ModelSelector

    organ_list = [o.strip() for o in args.organs.split(',') if o.strip()]
    if not organ_list:
        raise ValueError("--organs must contain at least one organ name")

    selector = ModelSelector()
    selector.build_index()
    plan = selector.select(
        organ_list,
        modality=args.modality,
        strategy=args.strategy,
        task_filter=args.task,
    )
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))


def cmd_info(args):
    """Execute info command"""
    from ..core import TaskRegistry
    
    task = TaskRegistry.get_task(args.task)
    
    print(f"Task: {task.name}")
    print("\nModels:")
    for model_name, model_info in task.models.items():
        print(f"  {model_name}:")
        print(f"    Task ID: {model_info.task_id}")
        print(f"    Modality: {model_info.modality}")
        print(f"    Description: {model_info.description}")
        print(f"    Labels: {model_info.labels}")
    
    print("\nInput requirements:")
    for key, value in task.input_requirements.items():
        print(f"  {key}: {value}")
    
    print("\nOutput config:")
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
                f"Invalid --model value '{entry}'. Expected format: DATASETID_MODELNAME=PATH"
            )
        key, raw_path = entry.split("=", 1)
        key = key.strip()
        raw_path = raw_path.strip()
        if not key or not raw_path:
            raise ValueError(
                f"Invalid --model value '{entry}'. Expected format: DATASETID_MODELNAME=PATH"
            )
        mappings[key] = raw_path
    return mappings


def cmd_download_models(args):
    """Execute download-models command"""
    from ..core import TaskRegistry

    model_names = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else None
    downloaded = TaskRegistry.download_task_models(
        task_name=args.task,
        model_names=model_names,
        force=args.force,
    )
    logger.info("Downloaded %d model(s) for task '%s'", len(downloaded), args.task)
    for model_name, path in downloaded.items():
        logger.info("  %s -> %s", model_name, path)


def cmd_install_models(args):
    """Execute install-models command"""
    from ..core import TaskRegistry

    if args.from_root:
        if args.model:
            raise ValueError("--model and --from are mutually exclusive")

        result = TaskRegistry.install_models_from_root(
            root=args.from_root,
            task_name=args.task,
            force=args.force,
            dry_run=args.dry_run,
        )

        planned = result["planned"]
        if args.dry_run:
            logger.info("Planned %d model(s) to install", len(planned))
        else:
            logger.info("Installed %d model(s)", len(result["installed"]))
        for model_name in sorted(planned):
            info = planned[model_name]
            logger.info(
                "  %s (%s / %s) <- %s", model_name, info["task"], info["task_id"], info["source"]
            )
        for source, prefix in result["ambiguous"]:
            logger.warning("Ambiguous candidate (prefix %s), skipped: %s", prefix, source)
        for source in result["unmatched"]:
            logger.warning("Unmatched candidate, skipped: %s", source)
        for source, task_id in result["skipped"]:
            logger.warning("Duplicate candidate for %s, skipped: %s", task_id, source)
        return

    if not args.model:
        raise ValueError("Provide either --model DATASETID_MODELNAME=PATH or --from ROOT")
    if not args.task:
        raise ValueError("--task is required with --model")
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
