#!/usr/bin/env python3
"""
Prepares OlmOCR checkpoints for deployment by:
1. Validating the model architecture
2. Copying model files to destination (disk or S3)
3. Downloading required tokenizer files from Hugging Face

Usage:
    python prepare_olmocr_checkpoint.py <source_path> <destination_path>
    
    source_path: Path to checkpoint (local or S3)
    destination_path: Where to save prepared checkpoint (local or S3)
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import tempfile

import boto3
import requests
from smart_open import smart_open
from tqdm import tqdm

from olmocr.s3_utils import parse_s3_path

# Hugging Face model ID for tokenizer files
HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
HF_BASE_URL = f"https://huggingface.co/{HF_MODEL_ID}/resolve/main"

# Required tokenizer files to download from Hugging Face
TOKENIZER_FILES = [
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
]

# Expected model architecture
EXPECTED_ARCHITECTURE = "Qwen2_5_VLForConditionalGeneration"

# Files to exclude from copying (training-related files)
EXCLUDED_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin"
}

s3_client = boto3.client("s3")


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def download_file_from_hf(filename: str, destination_dir: str) -> None:
    """Download a file from Hugging Face model repository."""
    url = f"{HF_BASE_URL}/{filename}"
    local_path = os.path.join(destination_dir, filename)
    
    print(f"Downloading {filename} from Hugging Face...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded {filename}")


def validate_checkpoint_architecture(config_path: str) -> None:
    """Validate that the checkpoint has the expected architecture."""
    print(f"Validating checkpoint architecture from {config_path}...")
    
    with smart_open(config_path, "r") as f:
        config_data = json.load(f)
    
    architectures = config_data.get("architectures", [])
    if EXPECTED_ARCHITECTURE not in architectures:
        raise ValueError(
            f"Invalid model architecture. Expected '{EXPECTED_ARCHITECTURE}' "
            f"but found: {architectures}"
        )
    
    print(f"✓ Valid architecture: {architectures}")


def copy_local_to_local(source_dir: str, dest_dir: str) -> None:
    """Copy files from local directory to local directory."""
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get list of files to copy
    files_to_copy = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file in EXCLUDED_FILES:
                print(f"Skipping excluded file: {file}")
                continue
            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, source_dir)
            files_to_copy.append((src_path, os.path.join(dest_dir, rel_path)))
    
    print(f"Copying {len(files_to_copy)} files from {source_dir} to {dest_dir}...")
    
    for src_path, dst_path in tqdm(files_to_copy, desc="Copying files"):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def download_file_from_s3(bucket: str, key: str, local_path: str) -> None:
    """Download a single file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)


def upload_file_to_s3(local_path: str, bucket: str, key: str) -> None:
    """Upload a single file to S3."""
    s3_client.upload_file(local_path, bucket, key)


def copy_s3_to_local(source_bucket: str, source_prefix: str, dest_dir: str) -> None:
    """Copy files from S3 to local directory."""
    os.makedirs(dest_dir, exist_ok=True)
    
    # List all objects in source
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)
    
    download_tasks = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            
            filename = os.path.basename(key)
            if filename in EXCLUDED_FILES:
                print(f"Skipping excluded file: {filename}")
                continue
            
            rel_path = os.path.relpath(key, source_prefix)
            local_path = os.path.join(dest_dir, rel_path)
            download_tasks.append((source_bucket, key, local_path))
    
    print(f"Downloading {len(download_tasks)} files from s3://{source_bucket}/{source_prefix} to {dest_dir}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_file_from_s3, bucket, key, local_path)
            for bucket, key, local_path in download_tasks
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            future.result()


def copy_local_to_s3(source_dir: str, dest_bucket: str, dest_prefix: str) -> None:
    """Copy files from local directory to S3."""
    # Get list of files to upload
    upload_tasks = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file in EXCLUDED_FILES:
                print(f"Skipping excluded file: {file}")
                continue
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, source_dir)
            s3_key = os.path.join(dest_prefix, rel_path)
            upload_tasks.append((local_path, dest_bucket, s3_key))
    
    print(f"Uploading {len(upload_tasks)} files from {source_dir} to s3://{dest_bucket}/{dest_prefix}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(upload_file_to_s3, local_path, bucket, key)
            for local_path, bucket, key in upload_tasks
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading"):
            future.result()


def copy_s3_to_s3(source_bucket: str, source_prefix: str, dest_bucket: str, dest_prefix: str) -> None:
    """Copy files from S3 to S3."""
    # List all objects in source
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)
    
    copy_tasks = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            
            filename = os.path.basename(key)
            if filename in EXCLUDED_FILES:
                print(f"Skipping excluded file: {filename}")
                continue
            
            rel_path = os.path.relpath(key, source_prefix)
            dest_key = os.path.join(dest_prefix, rel_path)
            copy_source = {"Bucket": source_bucket, "Key": key}
            copy_tasks.append((copy_source, dest_bucket, dest_key))
    
    print(f"Copying {len(copy_tasks)} files from s3://{source_bucket}/{source_prefix} to s3://{dest_bucket}/{dest_prefix}...")
    
    for copy_source, bucket, key in tqdm(copy_tasks, desc="Copying"):
        s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=key)


def prepare_checkpoint(source_path: str, dest_path: str) -> None:
    """Prepare OlmOCR checkpoint for deployment."""
    # First, validate the source checkpoint
    config_path = os.path.join(source_path, "config.json")
    if is_s3_path(source_path):
        config_path = f"{source_path}/config.json"
    
    validate_checkpoint_architecture(config_path)
    
    # Copy model files to destination
    print("\nCopying model files...")
    if is_s3_path(source_path) and is_s3_path(dest_path):
        # S3 to S3
        source_bucket, source_prefix = parse_s3_path(source_path)
        dest_bucket, dest_prefix = parse_s3_path(dest_path)
        copy_s3_to_s3(source_bucket, source_prefix, dest_bucket, dest_prefix)
    elif is_s3_path(source_path) and not is_s3_path(dest_path):
        # S3 to local
        source_bucket, source_prefix = parse_s3_path(source_path)
        copy_s3_to_local(source_bucket, source_prefix, dest_path)
    elif not is_s3_path(source_path) and is_s3_path(dest_path):
        # Local to S3
        dest_bucket, dest_prefix = parse_s3_path(dest_path)
        copy_local_to_s3(source_path, dest_bucket, dest_prefix)
    else:
        # Local to local
        copy_local_to_local(source_path, dest_path)
    
    # Download tokenizer files from Hugging Face
    print("\nDownloading tokenizer files from Hugging Face...")
    
    if is_s3_path(dest_path):
        # Download to temp directory first, then upload to S3
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(download_file_from_hf, filename, temp_dir)
                    for filename in TOKENIZER_FILES
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            
            # Upload to S3
            dest_bucket, dest_prefix = parse_s3_path(dest_path)
            upload_tasks = []
            for filename in TOKENIZER_FILES:
                local_path = os.path.join(temp_dir, filename)
                s3_key = os.path.join(dest_prefix, filename)
                upload_tasks.append((local_path, dest_bucket, s3_key))
            
            print("Uploading tokenizer files to S3...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(upload_file_to_s3, local_path, bucket, key)
                    for local_path, bucket, key in upload_tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    else:
        # Download directly to destination
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(download_file_from_hf, filename, dest_path)
                for filename in TOKENIZER_FILES
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    print(f"\n✓ Successfully prepared checkpoint at {dest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare OlmOCR checkpoint for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local to local
    python prepare_olmocr_checkpoint.py /path/to/checkpoint /path/to/output
    
    # S3 to S3
    python prepare_olmocr_checkpoint.py s3://bucket/checkpoint s3://bucket/prepared
    
    # S3 to local
    python prepare_olmocr_checkpoint.py s3://bucket/checkpoint /path/to/output
    
    # Local to S3
    python prepare_olmocr_checkpoint.py /path/to/checkpoint s3://bucket/prepared
        """
    )
    parser.add_argument("source", help="Source checkpoint path (local or S3)")
    parser.add_argument("destination", help="Destination path (local or S3)")
    
    args = parser.parse_args()
    
    try:
        prepare_checkpoint(args.source, args.destination)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())