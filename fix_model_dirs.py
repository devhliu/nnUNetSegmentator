#!/usr/bin/env python3
"""
Fix double-nested model directories
"""
from pathlib import Path
import shutil


def fix_directory(dir_path: Path):
    """
    Fix a directory that has a single subdirectory with the same name
    """
    if not dir_path.is_dir():
        return

    contents = list(dir_path.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        inner_dir = contents[0]
        print(f"Fixing {dir_path}: found inner dir {inner_dir.name}")
        
        # Move inner dir contents to temp
        temp_dir = dir_path.parent / f"temp_{dir_path.name}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        shutil.move(str(inner_dir), str(temp_dir))
        
        # Remove original dir if it still exists
        if dir_path.exists():
            shutil.rmtree(dir_path)
        
        # Rename temp to original
        shutil.move(str(temp_dir), str(dir_path))
        print(f"Fixed {dir_path}")


def main():
    # Fix lion models
    lion_root = Path.home() / ".nnunetsegmentator/models/lion"
    if lion_root.exists():
        print(f"Checking lion models at {lion_root}")
        for item in lion_root.iterdir():
            if item.is_dir():
                fix_directory(item)
    
    # Fix total_mr models
    total_mr_root = Path.home() / ".nnunetsegmentator/models/total_mr"
    if total_mr_root.exists():
        print(f"\nChecking total_mr models at {total_mr_root}")
        for item in total_mr_root.iterdir():
            if item.is_dir():
                fix_directory(item)


if __name__ == "__main__":
    main()
