"""
Configuration Management

This module provides configuration management for the nnUNet framework.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
import json
import yaml
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def get_default_model_root() -> Path:
    """
    Get the default model root directory.
    
    The model root path is determined by:
    1. NNUNETSEGMENTATOR_MODEL_ROOTPATH environment variable (if set)
    2. Default: ~/.nnunetsegmentator/models/
    
    Returns:
        Path to model root directory
    """
    env_path = os.environ.get('NNUNETSEGMENTATOR_MODEL_ROOTPATH')
    if env_path:
        return Path(env_path)
    return Path.home() / ".nnunetsegmentator" / "models"


def get_default_nnunet_workspace() -> Path:
    """
    Get the default nnUNet workspace directory.

    The workspace path is determined by:
    1. NNUNETSEGMENTATOR_NNUNET_WORKSPACE environment variable (if set)
    2. Default: ~/.nnunetsegmentator/nnunet_workspace/
    """
    env_path = os.environ.get("NNUNETSEGMENTATOR_NNUNET_WORKSPACE")
    if env_path:
        return Path(env_path)
    return Path.home() / ".nnunetsegmentator" / "nnunet_workspace"


@dataclass
class Config:
    """
    Configuration container for the nnUNet framework.
    
    This class manages all configuration settings including paths,
    model parameters, and processing options.
    
    Model Storage Path Structure:
        {NNUNETSEGMENTATOR_MODEL_ROOTPATH}/{task_name}/{task_id}
        
    Where:
        - NNUNETSEGMENTATOR_MODEL_ROOTPATH: Environment variable (default: ~/.nnunetsegmentator/models/)
        - task_name: Name of the segmentation task (e.g., 'gtrc', 'lion', 'total')
        - task_id: nnUNet task directory (e.g., 'Dataset881')
    """
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    model_dir: Path = field(default_factory=get_default_model_root)
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".nnunetsegmentator" / "cache")
    
    # Processing options
    num_workers: int = 1
    use_gpu: bool = True
    gpu_id: int = 0
    batch_size: int = 1
    
    # nnUNet specific
    nnunet_raw: Optional[Path] = None
    nnunet_preprocessed: Optional[Path] = None
    nnunet_results: Optional[Path] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        for attr in ['data_dir', 'output_dir', 'model_dir', 'cache_dir',
                     'nnunet_raw', 'nnunet_preprocessed', 'nnunet_results', 'log_file']:
            value = getattr(self, attr)
            if value is not None and isinstance(value, str):
                setattr(self, attr, Path(value))
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file (JSON or YAML)
            
        Returns:
            Config object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        logger.info(f"Loading configuration from: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config object
        """
        # Separate known and custom fields
        known_fields = {
            'data_dir', 'output_dir', 'model_dir', 'cache_dir',
            'num_workers', 'use_gpu', 'gpu_id', 'batch_size',
            'nnunet_raw', 'nnunet_preprocessed', 'nnunet_results',
            'log_level', 'log_file'
        }
        
        known_data = {k: v for k, v in data.items() if k in known_fields}
        custom_data = {k: v for k, v in data.items() if k not in known_fields}
        
        config = cls(**known_data)
        config.custom = custom_data
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        data = {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'model_dir': str(self.model_dir),
            'cache_dir': str(self.cache_dir),
            'num_workers': self.num_workers,
            'use_gpu': self.use_gpu,
            'gpu_id': self.gpu_id,
            'batch_size': self.batch_size,
            'log_level': self.log_level,
        }
        
        if self.nnunet_raw:
            data['nnunet_raw'] = str(self.nnunet_raw)
        if self.nnunet_preprocessed:
            data['nnunet_preprocessed'] = str(self.nnunet_preprocessed)
        if self.nnunet_results:
            data['nnunet_results'] = str(self.nnunet_results)
        if self.log_file:
            data['log_file'] = str(self.log_file)
        
        data.update(self.custom)
        
        return data
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Output file path (JSON or YAML)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(filepath, 'w') as f:
            if filepath.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")
    
    def update(self, **kwargs) -> 'Config':
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration key-value pairs
            
        Returns:
            Updated Config object
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom[key] = value
        
        return self
    
    def setup_nnunet_environment(self) -> None:
        """
        Setup nnUNet environment variables.
        
        This method sets the required nnUNet environment variables
        based on the configuration.
        """
        import os

        workspace_root = get_default_nnunet_workspace()
        nnunet_raw = self.nnunet_raw or Path(os.environ.get('nnUNet_raw', workspace_root / "raw"))
        nnunet_preprocessed = self.nnunet_preprocessed or Path(
            os.environ.get('nnUNet_preprocessed', workspace_root / "preprocessed")
        )
        nnunet_results = self.nnunet_results or Path(
            os.environ.get('nnUNet_results', workspace_root / "results")
        )

        model_root = self.model_dir.expanduser().resolve()
        for name, path in (
            ("nnUNet_raw", nnunet_raw),
            ("nnUNet_preprocessed", nnunet_preprocessed),
            ("nnUNet_results", nnunet_results),
        ):
            resolved = path.expanduser().resolve()
            if resolved == model_root or model_root in resolved.parents:
                raise ValueError(
                    f"{name} must not be inside model_dir ({model_root}). "
                    "Use a separate nnUNet workspace path."
                )

        nnunet_raw.mkdir(parents=True, exist_ok=True)
        nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
        nnunet_results.mkdir(parents=True, exist_ok=True)

        self.nnunet_raw = nnunet_raw
        self.nnunet_preprocessed = nnunet_preprocessed
        self.nnunet_results = nnunet_results

        os.environ['nnUNet_raw'] = str(self.nnunet_raw)
        os.environ['nnUNet_preprocessed'] = str(self.nnunet_preprocessed)
        os.environ['nnUNet_results'] = str(self.nnunet_results)

        logger.debug(f"Set nnUNet_raw = {self.nnunet_raw}")
        logger.debug(f"Set nnUNet_preprocessed = {self.nnunet_preprocessed}")
        logger.debug(f"Set nnUNet_results = {self.nnunet_results}")
    
    def setup_logging(self) -> None:
        """
        Setup logging based on configuration.
        """
        import sys
        
        # Set log level
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        # Add file handler if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logging.getLogger().addHandler(file_handler)
    
    def ensure_directories(self) -> None:
        """
        Create all configured directories if they don't exist.
        """
        for attr in ['data_dir', 'output_dir', 'model_dir', 'cache_dir']:
            path = getattr(self, attr)
            if path:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory: {path}")
    
    def get_model_path(self, task_name: str, task_id: str = None) -> Path:
        """
        Get the model path following the standard structure.
        
        Model path structure:
            {model_dir}/{task_name}/{task_id}
        
        Args:
            task_name: Name of the segmentation task (e.g., 'gtrc', 'lion', 'total')
            task_id: Optional task ID directory name (e.g., 'Dataset881')
                    If None, returns the task directory
        
        Returns:
            Path to model directory
        
        Examples:
            >>> config = Config()
            >>> config.get_model_path('gtrc', 'Dataset881_gtrc_psma')
            Path('/home/user/.nnunetsegmentator/models/gtrc/Dataset881_gtrc_psma')
            
            >>> config.get_model_path('lion')
            Path('/home/user/.nnunetsegmentator/models/lion')
        """
        if task_id:
            return self.model_dir / task_name / task_id
        return self.model_dir / task_name
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config(data_dir={self.data_dir}, output_dir={self.output_dir}, model_dir={self.model_dir}, ...)"
