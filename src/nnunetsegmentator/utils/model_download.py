"""
Model Download and Weight Management Utilities

This module provides functionality for downloading, verifying, and managing
pretrained model weights from various sources (Zenodo, HuggingFace, etc.).
"""

import hashlib
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError
import json
import zipfile
import tarfile

from .model_layout import is_downloadable_url

logger = logging.getLogger(__name__)


class ModelDownloader:
    """
    Handles downloading and verification of pretrained model weights.
    """
    
    # Model registry URLs
    ZENODO_API = "https://zenodo.org/api/"
    HUGGINGFACE_API = "https://huggingface.co/api/"
    
    # Default model storage location
    DEFAULT_MODEL_DIR = Path.home() / ".nnunetsegmentator" / "models"
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the model downloader.
        
        Args:
            model_dir: Directory to store downloaded models
        """
        self.model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index file for tracking downloads
        self.index_file = self.model_dir / "model_index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load model index from file."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save model index to file."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def compute_checksum(self, filepath: Path, algorithm: str = "sha256") -> str:
        """
        Compute checksum of a file.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm (md5, sha256)
        
        Returns:
            Hexadecimal checksum string
        """
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def verify_checksum(self, filepath: Path, expected: str, algorithm: str = "sha256") -> bool:
        """
        Verify file checksum.
        
        Args:
            filepath: Path to file
            expected: Expected checksum
            algorithm: Hash algorithm
        
        Returns:
            True if checksum matches
        """
        actual = self.compute_checksum(filepath, algorithm)
        if actual != expected:
            logger.warning(f"Checksum mismatch for {filepath}: expected {expected}, got {actual}")
            return False
        return True
    
    def download_file(self, url: str, destination: Path, 
                     show_progress: bool = True) -> Path:
        """
        Download a file from URL.
        
        Args:
            url: Source URL
            destination: Destination path
            show_progress: Show download progress
        
        Returns:
            Path to downloaded file
        """
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {url} to {destination}")
        
        try:
            if show_progress:
                # Custom reporthook for progress
                def reporthook(count, block_size, total_size):
                    if total_size > 0:
                        percent = int(count * block_size * 100 / total_size)
                        logger.debug("Downloading %s: %s%%", destination.name, percent)
                
                urlretrieve(url, destination, reporthook)
            else:
                urlretrieve(url, destination)
            
            logger.info(f"Download complete: {destination}")
            return destination
            
        except (URLError, HTTPError) as e:
            logger.error(f"Failed to download {url}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    def download_from_zenodo(self, record_id: str, filename: str,
                            destination: Optional[Path] = None) -> Path:
        """
        Download model from Zenodo.
        
        Args:
            record_id: Zenodo record ID
            filename: Name of file to download
            destination: Destination directory
        
        Returns:
            Path to downloaded file
        """
        # Get record metadata
        api_url = f"{self.ZENODO_API}records/{record_id}"
        
        try:
            with urlopen(api_url) as response:
                metadata = json.loads(response.read().decode())
        except (URLError, HTTPError, json.JSONDecodeError) as e:
            logger.error(f"Failed to fetch Zenodo metadata: {e}")
            raise RuntimeError(f"Failed to fetch Zenodo metadata for record {record_id}") from e
        
        # Find file in record
        files = metadata.get("files", [])
        file_info = None
        for f in files:
            if f["key"] == filename:
                file_info = f
                break
        
        if file_info is None:
            raise ValueError(f"File {filename} not found in Zenodo record {record_id}")
        
        # Download file
        download_url = file_info["links"]["self"]
        checksum = file_info.get("checksum", "").split(":")[-1]  # Remove algorithm prefix
        
        if destination is None:
            destination = self.model_dir / filename
        
        filepath = self.download_file(download_url, destination)
        
        # Verify checksum if available
        if checksum:
            if not self.verify_checksum(filepath, checksum, "md5"):
                logger.warning("Checksum verification failed, file may be corrupted")
        
        return filepath
    
    def extract_archive(self, archive_path: Path, extract_dir: Optional[Path] = None) -> Path:
        """
        Extract archive file.
        
        Args:
            archive_path: Path to archive
            extract_dir: Extraction directory
        
        Returns:
            Path to extracted directory
        """
        archive_path = Path(archive_path)
        
        temp_extract_dir = None
        if extract_dir is None:
            temp_extract_dir = archive_path.parent / archive_path.stem
            extract_dir = temp_extract_dir
        else:
            temp_extract_dir = archive_path.parent / f"temp_{archive_path.stem}"
        
        temp_extract_dir = Path(temp_extract_dir)
        extract_dir = Path(extract_dir)
        
        logger.info(f"Extracting {archive_path} to {temp_extract_dir}")
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                self._safe_extract_zip(zf, temp_extract_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tf:
                self._safe_extract_tar(tf, temp_extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        # Check if the archive has a single top-level directory
        contents = list(temp_extract_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            # If single directory, move its contents to extract_dir
            single_dir = contents[0]
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            shutil.move(str(single_dir), str(extract_dir))
            # Clean up temp directory
            if temp_extract_dir.exists() and temp_extract_dir != extract_dir:
                shutil.rmtree(temp_extract_dir)
        else:
            # Otherwise, just move the temp dir to extract_dir
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            if temp_extract_dir != extract_dir:
                shutil.move(str(temp_extract_dir), str(extract_dir))
        
        logger.info(f"Extraction complete: {extract_dir}")
        return extract_dir

    def _safe_extract_zip(self, archive: zipfile.ZipFile, destination: Path) -> None:
        """
        Extract zip archive while preventing path traversal attacks.
        """
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not str(member_path).startswith(str(destination.resolve())):
                raise RuntimeError(f"Unsafe archive member path detected: {member.filename}")
        archive.extractall(destination)

    def _safe_extract_tar(self, archive: tarfile.TarFile, destination: Path) -> None:
        """
        Extract tar archive while preventing path traversal attacks.
        """
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination.resolve())):
                raise RuntimeError(f"Unsafe archive member path detected: {member.name}")
        archive.extractall(destination)
    
    def download_model(
        self,
        model_name: str,
        url: str,
        checksum: Optional[str] = None,
        force: bool = False,
        target_dir_name: Optional[str] = None,
    ) -> Path:
        """
        Download and prepare a model.
        
        Args:
            model_name: Name of model (used for index key)
            url: Download URL
            checksum: Expected checksum
            force: Force re-download
            target_dir_name: Optional target directory name for extracted model
        
        Returns:
            Path to model directory
        """
        target_name = target_dir_name or model_name
        model_path = self.model_dir / target_name
        
        # Check if already downloaded
        if not force and model_name in self.index:
            indexed_path = Path(self.index[model_name]["path"])
            if indexed_path.exists():
                logger.info(f"Model {model_name} already downloaded at {indexed_path}")
                return indexed_path
        
        # Download
        if not is_downloadable_url(url):
            raise ValueError(
                f"Model URL for '{model_name}' is not publicly downloadable ('{url}'). "
                "Provide valid weights, register a local model path, or use "
                "`install-models` with a local weights root."
            )

        suffix = Path(url.split("?")[0]).suffix or ".zip"
        archive_path = self.model_dir / f"{model_name}{suffix}"
        self.download_file(url, archive_path)
        
        # Verify checksum
        if checksum:
            if not self.verify_checksum(archive_path, checksum):
                raise RuntimeError("Checksum verification failed")
        
        # Extract
        model_path = self.extract_archive(archive_path, model_path)
        
        # Clean up archive
        archive_path.unlink()
        
        # Update index
        self.index[model_name] = {
            "path": str(model_path),
            "url": url,
            "checksum": checksum,
        }
        self._save_index()
        
        return model_path
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get path to downloaded model.
        
        Args:
            model_name: Name of model
        
        Returns:
            Path to model or None if not found
        """
        if model_name in self.index:
            path = Path(self.index[model_name]["path"])
            if path.exists():
                return path
        
        # Check default location
        model_path = self.model_dir / model_name
        if model_path.exists():
            return model_path
        
        return None
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all downloaded models.
        
        Returns:
            Dictionary of model information
        """
        models = {}
        for model_name, info in self.index.items():
            path = Path(info["path"])
            if path.exists():
                models[model_name] = {
                    **info,
                    "exists": True,
                    "size": sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                }
            else:
                models[model_name] = {**info, "exists": False}
        
        return models
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a downloaded model.
        
        Args:
            model_name: Name of model
        
        Returns:
            True if model was removed
        """
        if model_name not in self.index:
            return False
        
        path = Path(self.index[model_name]["path"])
        if path.exists():
            shutil.rmtree(path)
        
        del self.index[model_name]
        self._save_index()
        
        logger.info(f"Removed model: {model_name}")
        return True
