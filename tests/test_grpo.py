#!/usr/bin/env python3
"""
Test suite for GRPO training dataloader.
Tests the OlmOCRDataset class and its functionality with Olmocr-bench format.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmocr.train.grpo_train import OlmOCRDataset, collate_fn


class TestGRPODataloader(unittest.TestCase):
    """Test cases for the GRPO dataloader."""
    
    @classmethod
    def setUpClass(cls):
        """Create a temporary bench_data folder with test data."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.bench_data_folder = cls.temp_dir
        cls.pdfs_folder = os.path.join(cls.bench_data_folder, "pdfs")
        
        # Create folder structure
        os.makedirs(os.path.join(cls.pdfs_folder, "test_pdfs"), exist_ok=True)
        
        # Create dummy PDF files
        cls.pdf_files = []
        for i in range(3):
            pdf_path = os.path.join(cls.pdfs_folder, "test_pdfs", f"test_{i}.pdf")
            # Create a minimal valid PDF
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4\n%%EOF")
            cls.pdf_files.append(pdf_path)
        
        # Create test JSONL files
        cls.jsonl_file1 = os.path.join(cls.bench_data_folder, "test1.jsonl")
        cls.jsonl_file2 = os.path.join(cls.bench_data_folder, "test2.jsonl")
        
        # Write test data to JSONL files
        test_data1 = [
            {"pdf": "test_pdfs/test_0.pdf", "page": 0, "id": "test_0_001", "type": "math", "math": "x + y = z"},
            {"pdf": "test_pdfs/test_0.pdf", "page": 0, "id": "test_0_002", "type": "text", "text": "Sample text"},
            {"pdf": "test_pdfs/test_1.pdf", "page": 0, "id": "test_1_001", "type": "math", "math": "a^2 + b^2 = c^2"},
            {"pdf": "test_pdfs/test_1.pdf", "page": 1, "id": "test_1_002", "type": "text", "text": "Another sample"},
        ]
        
        test_data2 = [
            {"pdf": "test_pdfs/test_2.pdf", "page": 0, "id": "test_2_001", "type": "table", "table": "col1,col2"},
            {"pdf": "test_pdfs/test_0.pdf", "page": 0, "id": "test_0_003", "type": "text", "text": "More text"},
            {"pdf": "test_pdfs/test_2.pdf", "page": 0, "id": "test_2_002", "type": "math", "math": "\\int_0^1 x dx"},
        ]
        
        with open(cls.jsonl_file1, "w") as f:
            for entry in test_data1:
                f.write(json.dumps(entry) + "\n")
        
        with open(cls.jsonl_file2, "w") as f:
            for entry in test_data2:
                f.write(json.dumps(entry) + "\n")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        shutil.rmtree(cls.temp_dir)
    
    def test_dataset_initialization(self):
        """Test that dataset initializes correctly."""
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=None,
            target_longest_image_dim=1024,
        )
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.bench_data_folder, self.bench_data_folder)
        self.assertEqual(dataset.pdf_folder, self.pdfs_folder)
        self.assertTrue(len(dataset) > 0)
    
    def test_unique_pdf_loading(self):
        """Test that unique PDFs are loaded correctly."""
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=None,
            target_longest_image_dim=1024,
        )
        
        # Should have 4 unique PDF+page combinations:
        # test_0.pdf page 0, test_1.pdf page 0, test_1.pdf page 1, test_2.pdf page 0
        self.assertEqual(len(dataset), 4)
        
        # Check that samples have correct structure
        for sample in dataset.samples:
            self.assertIn("pdf_path", sample)
            self.assertIn("pdf_name", sample)
            self.assertIn("page", sample)
            self.assertIn("jsonl_file", sample)
            self.assertIn("test_ids", sample)
            self.assertIn("entries", sample)
    
    def test_test_id_aggregation(self):
        """Test that test IDs are correctly aggregated per PDF+page."""
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=None,
            target_longest_image_dim=1024,
        )
        
        # Find the sample for test_0.pdf page 0
        test_0_sample = None
        for sample in dataset.samples:
            if "test_0.pdf" in sample["pdf_name"] and sample["page"] == 0:
                test_0_sample = sample
                break
        
        self.assertIsNotNone(test_0_sample)
        # Should have 3 test IDs for test_0.pdf page 0
        self.assertEqual(len(test_0_sample["test_ids"]), 3)
        self.assertIn("test_0_001", test_0_sample["test_ids"])
        self.assertIn("test_0_002", test_0_sample["test_ids"])
        self.assertIn("test_0_003", test_0_sample["test_ids"])
    
    def test_max_samples_limit(self):
        """Test that max_samples correctly limits the dataset size."""
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=2,
            target_longest_image_dim=1024,
        )
        
        self.assertEqual(len(dataset), 2)
    
    @patch('olmocr.train.grpo_train.render_pdf_to_base64png')
    @patch('olmocr.train.grpo_train.build_no_anchoring_v4_yaml_prompt')
    def test_getitem_format(self, mock_prompt, mock_render):
        """Test that __getitem__ returns the correct format."""
        # Mock the rendering and prompt functions
        mock_render.return_value = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # 1x1 white pixel PNG
        mock_prompt.return_value = "Test prompt"
        
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=1,
            target_longest_image_dim=1024,
        )
        
        item = dataset[0]
        
        self.assertIsNotNone(item)
        self.assertIn("prompt", item)
        self.assertIn("pdf_path", item)
        self.assertIn("jsonl_file", item)
        self.assertIn("test_ids", item)
        self.assertIn("image", item)
        
        # Check prompt structure
        self.assertIsInstance(item["prompt"], list)
        self.assertEqual(len(item["prompt"]), 1)
        self.assertEqual(item["prompt"][0]["role"], "user")
        self.assertIsInstance(item["prompt"][0]["content"], list)
        self.assertEqual(len(item["prompt"][0]["content"]), 2)
        
        # Check other fields
        self.assertIsInstance(item["pdf_path"], str)
        self.assertIsInstance(item["jsonl_file"], str)
        self.assertIsInstance(item["test_ids"], list)
        self.assertTrue(len(item["test_ids"]) > 0)
    
    @patch('olmocr.train.grpo_train.render_pdf_to_base64png')
    @patch('olmocr.train.grpo_train.build_no_anchoring_v4_yaml_prompt')
    def test_collate_function(self, mock_prompt, mock_render):
        """Test that the collate function works correctly."""
        # Mock the rendering and prompt functions
        mock_render.return_value = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        mock_prompt.return_value = "Test prompt"
        
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=2,
            target_longest_image_dim=1024,
        )
        
        # Create a batch
        batch = [dataset[0], dataset[1]]
        collated = collate_fn(batch)
        
        self.assertIsNotNone(collated)
        self.assertIn("prompts", collated)
        self.assertIn("images", collated)
        self.assertIn("pdf_paths", collated)
        self.assertIn("jsonl_files", collated)
        self.assertIn("test_ids", collated)
        
        # Check batch size consistency
        self.assertEqual(len(collated["prompts"]), 2)
        self.assertEqual(len(collated["images"]), 2)
        self.assertEqual(len(collated["pdf_paths"]), 2)
        self.assertEqual(len(collated["jsonl_files"]), 2)
        self.assertEqual(len(collated["test_ids"]), 2)
    
    def test_collate_with_none_values(self):
        """Test that collate function handles None values correctly."""
        batch = [None, {"prompt": [], "image": None, "pdf_path": "test.pdf", 
                        "jsonl_file": "test.jsonl", "test_ids": ["id1"]}, None]
        
        collated = collate_fn(batch)
        
        self.assertIsNotNone(collated)
        self.assertEqual(len(collated["prompts"]), 1)
    
    def test_empty_jsonl_handling(self):
        """Test handling of empty JSONL files."""
        # Create an empty JSONL file
        empty_jsonl = os.path.join(self.bench_data_folder, "empty.jsonl")
        open(empty_jsonl, "w").close()
        
        # Should still work with other non-empty files
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=None,
            target_longest_image_dim=1024,
        )
        
        self.assertTrue(len(dataset) > 0)
        
        # Clean up
        os.remove(empty_jsonl)
    
    def test_malformed_jsonl_handling(self):
        """Test handling of malformed JSONL entries."""
        # Create a JSONL with some malformed entries
        malformed_jsonl = os.path.join(self.bench_data_folder, "malformed.jsonl")
        with open(malformed_jsonl, "w") as f:
            f.write('{"pdf": "test.pdf", "id": "valid_1"}\n')
            f.write('not valid json\n')
            f.write('{"pdf": "test2.pdf", "id": "valid_2"}\n')
        
        # Should skip malformed entries but process valid ones
        dataset = OlmOCRDataset(
            bench_data_folder=self.bench_data_folder,
            processor=None,
            max_samples=None,
            target_longest_image_dim=1024,
        )
        
        # Should still have entries from valid files
        self.assertTrue(len(dataset) > 0)
        
        # Clean up
        os.remove(malformed_jsonl)
    
    def test_missing_pdf_folder(self):
        """Test error handling when pdfs folder is missing."""
        temp_bad_folder = tempfile.mkdtemp()
        
        with self.assertRaises(ValueError) as context:
            dataset = OlmOCRDataset(
                bench_data_folder=temp_bad_folder,
                processor=None,
                max_samples=None,
                target_longest_image_dim=1024,
            )
        
        self.assertIn("PDFs folder not found", str(context.exception))
        
        # Clean up
        shutil.rmtree(temp_bad_folder)
    
    def test_no_jsonl_files(self):
        """Test error handling when no JSONL files are present."""
        temp_folder = tempfile.mkdtemp()
        os.makedirs(os.path.join(temp_folder, "pdfs"))
        
        with self.assertRaises(ValueError) as context:
            dataset = OlmOCRDataset(
                bench_data_folder=temp_folder,
                processor=None,
                max_samples=None,
                target_longest_image_dim=1024,
            )
        
        self.assertIn("No JSONL files found", str(context.exception))
        
        # Clean up
        shutil.rmtree(temp_folder)


class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests with real bench data if available."""
    
    @unittest.skipUnless(
        os.path.exists("/home/ubuntu/olmocr/olmOCR-bench/bench_data"),
        "Real bench data not available"
    )
    def test_with_real_bench_data(self):
        """Test with real bench data if available."""
        bench_data_folder = "/home/ubuntu/olmocr/olmOCR-bench/bench_data"
        
        dataset = OlmOCRDataset(
            bench_data_folder=bench_data_folder,
            processor=None,
            max_samples=5,
            target_longest_image_dim=1024,
        )
        
        self.assertEqual(len(dataset), 5)
        
        # Test that we can iterate through the dataset
        for i in range(len(dataset)):
            item = dataset[i]
            if item is not None:  # Some PDFs might fail to render
                self.assertIn("prompt", item)
                self.assertIn("pdf_path", item)
                self.assertIn("jsonl_file", item)
                self.assertIn("test_ids", item)
                
                # Verify paths exist
                self.assertTrue(os.path.exists(item["pdf_path"]))
                self.assertTrue(os.path.exists(item["jsonl_file"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)