#!/usr/bin/env python3
"""
Script to selectively copy .md and .pdf files from a source directory to an output directory
based on whether the .md files contain LaTeX math commands.
"""

import argparse
import shutil
from pathlib import Path
from typing import List

# LaTeX commands that indicate math content
LATEX_COMMANDS = [
    # Lists & basic content
    r"\begin{itemize}",
    r"\begin{enumerate}",
    r"\item",
    # Figures, tables, and captions
    r"\begin{figure}",
    r"\includegraphics",
    r"\caption",
    r"\label",
    r"\ref",
    r"\eqref",
    r"\begin{table}",
    r"\begin{tabular}",
    # Formatting
    r"\textit",
    r"\textbb",
    # Math (strong signals)
    r"\begin{equation}",
    r"\begin{align}",
    r"\frac",
    r"\sum",
    r"\int",
    r"\sqrt",
    r"\prod",
    r"\lim",
    r"\binom",
    r"\mathbb",
    r"\mathcal",
    r"\to",
    r"\varphi",
    r"\cdot",
    r"\langle",
    r"\rangle",
    # Citations (bibliography stacks)
    r"\cite",
]


def contains_latex_math(file_path: Path) -> bool:
    """
    Check if a markdown file contains any LaTeX math commands.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        True if the file contains LaTeX math, False otherwise
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        for command in LATEX_COMMANDS:
            if command in content:
                return True
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def copy_files_with_math(source_dir: Path, output_dir: Path, verbose: bool = False) -> None:
    """
    Recursively copy .md and corresponding .pdf files that contain LaTeX math.
    
    Args:
        source_dir: Source directory containing .md and .pdf files
        output_dir: Output directory where filtered files will be copied
        verbose: Whether to print verbose output
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_md_files = 0
    copied_files = 0
    skipped_files = 0
    
    # Find all markdown files recursively
    md_files = list(source_dir.rglob("*.md"))
    total_md_files = len(md_files)
    
    print(f"Found {total_md_files} markdown files in {source_dir}")
    
    for md_file in md_files:
        # Check if the markdown file contains LaTeX math
        if contains_latex_math(md_file):
            # Get the relative path from source directory
            rel_path = md_file.relative_to(source_dir)
            
            # Create the same directory structure in output
            output_md_path = output_dir / rel_path
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the markdown file
            shutil.copy2(md_file, output_md_path)
            
            # Look for corresponding PDF file
            pdf_file = md_file.with_suffix('.pdf')
            if pdf_file.exists():
                output_pdf_path = output_md_path.with_suffix('.pdf')
                shutil.copy2(pdf_file, output_pdf_path)
                
                if verbose:
                    print(f"Copied: {rel_path} and corresponding PDF")
            else:
                if verbose:
                    print(f"Copied: {rel_path} (no PDF found)")
            
            copied_files += 1
        else:
            skipped_files += 1
            if verbose:
                print(f"Skipped: {md_file.relative_to(source_dir)} (no LaTeX math)")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total markdown files: {total_md_files}")
    print(f"  Files with LaTeX math (copied): {copied_files}")
    print(f"  Files without LaTeX math (skipped): {skipped_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy .md and .pdf files containing LaTeX math from source to output directory"
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="Source directory containing .md and .pdf files"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory where filtered files will be copied"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    # Validate source directory
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return 1
    
    if not source_dir.is_dir():
        print(f"Error: '{source_dir}' is not a directory")
        return 1
    
    # Run the copy operation
    copy_files_with_math(source_dir, output_dir, verbose=args.verbose)
    
    return 0


if __name__ == "__main__":
    exit(main())