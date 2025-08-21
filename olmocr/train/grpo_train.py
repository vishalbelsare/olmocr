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
import wandb
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

def simple_length_reward(completions_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of token count)."""
    return [float(len(ids)) for ids in completions_ids]


def main():
    parser = argparse.ArgumentParser(description="GRPO training for OlmOCR")
    parser.add_argument(
        "--train_bench_data_folder", 
        type=str, 
        required=True,
        help="Path to training bench data folder containing JSONL files and pdfs subfolder"
    )
    parser.add_argument(
        "--eval_bench_data_folder", 
        type=str, 
        required=False,
        default=None,
        help="Path to evaluation bench data folder (optional, uses train folder if not specified)"
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
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (default: use all)"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples to use (default: use all)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="olmocr-grpo",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Set up output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    logger.info(f"Initialized wandb project: {args.wandb_project}")
    report_to = ["wandb"]

    
    # Verify train bench_data_folder exists
    if not os.path.exists(args.train_bench_data_folder):
        logger.error(f"Train bench data folder not found: {args.train_bench_data_folder}")
        return
    
    # Set eval folder to train folder if not specified
    if args.eval_bench_data_folder is None:
        args.eval_bench_data_folder = args.train_bench_data_folder
        logger.info(f"Using train folder for evaluation: {args.eval_bench_data_folder}")
    elif not os.path.exists(args.eval_bench_data_folder):
        logger.error(f"Eval bench data folder not found: {args.eval_bench_data_folder}")
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
    
    # Create training dataset
    logger.info(f"Creating training dataset from: {args.train_bench_data_folder}")
    train_dataset = OlmOCRDataset(
        bench_data_folder=args.train_bench_data_folder,
        processor=processor,
        max_samples=args.max_train_samples,
        target_longest_image_dim=1288,
    )
    
    if len(train_dataset) == 0:
        logger.error("No samples found in training dataset!")
        return
    
    # Create evaluation dataset
    logger.info(f"Creating evaluation dataset from: {args.eval_bench_data_folder}")
    eval_dataset = OlmOCRDataset(
        bench_data_folder=args.eval_bench_data_folder,
        processor=processor,
        max_samples=args.max_eval_samples,
        target_longest_image_dim=1288,
    )
    
    if len(eval_dataset) == 0:
        logger.warning("No samples found in evaluation dataset, using training dataset for eval")
        eval_dataset = train_dataset
    
    # Set up GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_steps=10,
        max_prompt_length=3000,
        max_completion_length=3000,
        temperature=0.7,
        report_to=report_to,
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=0,
    )
    
    # Initialize GRPO trainer
    logger.info("Initializing GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=simple_length_reward,
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
        
        # Close wandb if it was used
        if args.use_wandb:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()