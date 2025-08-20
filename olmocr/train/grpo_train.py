"""
GRPO (Generative Reward-based Policy Optimization) training script for OlmOCR.
"""

import argparse
import logging
import os
from typing import List, Dict, Any, Optional, Set
import asyncio
import json
import random
from pathlib import Path
import glob

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)
from trl import GRPOConfig, GRPOTrainer
from PIL import Image
import base64
from io import BytesIO

from olmocr.train.config import Config
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class OlmOCRDataset(Dataset):
    """Dataset for loading PDF pages from Olmocr-bench format JSONL files."""
    
    def __init__(
        self,
        bench_data_folder: str,
        processor,
        max_samples: Optional[int] = None,
        target_longest_image_dim: int = 1024,
    ):
        self.bench_data_folder = bench_data_folder
        self.processor = processor
        self.target_longest_image_dim = target_longest_image_dim
        self.max_samples = max_samples
        
        # Find PDF folder
        self.pdf_folder = os.path.join(bench_data_folder, "pdfs")
        if not os.path.exists(self.pdf_folder):
            raise ValueError(f"PDFs folder not found at {self.pdf_folder}")
        
        # Load unique PDFs from JSONL files
        self.samples = self._load_unique_pdfs_from_jsonl()
        
        logger.info(f"Created dataset with {len(self.samples)} unique PDF samples")
    
    def _load_unique_pdfs_from_jsonl(self) -> List[Dict[str, Any]]:
        """Load unique PDFs from JSONL files in the bench_data folder, tracking all test cases per PDF."""
        jsonl_files = glob.glob(os.path.join(self.bench_data_folder, "*.jsonl"))
        
        if not jsonl_files:
            raise ValueError(f"No JSONL files found in {self.bench_data_folder}")
        
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        
        # Track unique PDFs and their test cases
        pdf_data: Dict[str, Dict[str, Any]] = {}
        
        for jsonl_file in jsonl_files:
            logger.info(f"Processing {os.path.basename(jsonl_file)}")
            
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        pdf_name = entry.get("pdf")
                        page = entry.get("page", 0)
                        test_id = entry.get("id")
                        
                        if pdf_name and test_id:
                            # Create unique key for PDF+page combination
                            pdf_page_key = f"{pdf_name}::{page}"
                            
                            if pdf_page_key not in pdf_data:
                                # First time seeing this PDF+page
                                pdf_path = os.path.join(self.pdf_folder, pdf_name)
                                pdf_data[pdf_page_key] = {
                                    "pdf_path": pdf_path,
                                    "pdf_name": pdf_name,
                                    "page": page,
                                    "jsonl_file": jsonl_file,
                                    "test_ids": [test_id],
                                    "entries": [entry]
                                }
                            else:
                                # Add test case to existing PDF+page
                                pdf_data[pdf_page_key]["test_ids"].append(test_id)
                                pdf_data[pdf_page_key]["entries"].append(entry)
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing entry in {jsonl_file}: {e}")
                        continue
        
        # Convert to list and apply max_samples limit
        samples = list(pdf_data.values())
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pdf_path = sample["pdf_path"]
        page_num = sample["page"]
        jsonl_file = sample["jsonl_file"]
        test_ids = sample["test_ids"]
        
        try:
            # Render PDF page to base64 image
            image_base64 = render_pdf_to_base64png(
                pdf_path, 
                page_num, 
                target_longest_image_dim=self.target_longest_image_dim
            )
            
            # Convert base64 to PIL Image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
            # Build the text prompt
            text_prompt = build_no_anchoring_v4_yaml_prompt()
            
            # Create messages in the format expected by Qwen2-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image"},
                    ],
                }
            ]
            
            # Return the required format
            return {
                "prompt": messages,
                "pdf_path": pdf_path,
                "jsonl_file": jsonl_file,
                "test_ids": test_ids,
                "image": image,  # Include the PIL image for processing later
            }
            
        except Exception as e:
            logger.error(f"Failed to process sample {idx}: {e}")
            # Return None if processing fails
            return None


def collate_fn(batch):
    """Custom collate function to handle the new batch format with prompts and metadata."""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if not batch:
        return None
    
    # Collect all components
    prompts = [item["prompt"] for item in batch]
    images = [item["image"] for item in batch]
    pdf_paths = [item["pdf_path"] for item in batch]
    jsonl_files = [item["jsonl_file"] for item in batch]
    test_ids = [item["test_ids"] for item in batch]
    
    # Return batch with all required information
    return {
        "prompts": prompts,
        "images": images, 
        "pdf_paths": pdf_paths,
        "jsonl_files": jsonl_files,
        "test_ids": test_ids,
    }


def simple_length_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Simple reward function that rewards completions close to 100 tokens.
    Returns higher rewards for completions closer to the target length.
    """
    target_length = 100
    rewards = []
    
    for completion in completions:
        # Count tokens (simple word-based approximation)
        tokens = completion.split()
        length = len(tokens)
        
        # Calculate reward based on distance from target
        distance = abs(length - target_length)
        
        # Reward function: max reward of 1.0 at target length, 
        # decreasing as we get further away
        if distance == 0:
            reward = 1.0
        else:
            # Exponential decay based on distance
            reward = max(0.0, 1.0 - (distance / target_length))
        
        rewards.append(reward)
        
    logger.info(f"Reward stats: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO training for OlmOCR")
    parser.add_argument(
        "--bench_data_folder", 
        type=str, 
        required=True,
        help="Path to bench data folder containing JSONL files and pdfs subfolder"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model checkpoint to load"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/grpo_test",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to use (for testing)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify bench_data_folder exists
    if not os.path.exists(args.bench_data_folder):
        logger.error(f"Bench data folder not found: {args.bench_data_folder}")
        return
    
    # Load processor
    logger.info(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    if "Qwen2.5-VL" in args.model_name:
        model_class = Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in args.model_name:
        model_class = Qwen2VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")
    
    model = model_class.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Create dataset from bench data folder
    logger.info(f"Creating dataset from bench data folder: {args.bench_data_folder}")
    dataset = OlmOCRDataset(
        bench_data_folder=args.bench_data_folder,
        processor=processor,
        max_samples=args.max_samples,
        target_longest_image_dim=1024,
    )
    
    if len(dataset) == 0:
        logger.error("No samples found in dataset!")
        return
    
    # Set up GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_steps=10,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        torch_dtype=torch.bfloat16,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    
    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=processor,
        train_dataset=dataset,
        reward_function=simple_length_reward,
        data_collator=collate_fn,
    )
    
    # Start training
    logger.info("Starting GRPO training")
    try:
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {args.output_dir}")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()