#!/usr/bin/env python
"""Tests for dataloader pipeline steps."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from olmocr.prompts.prompts import PageResponse
from olmocr.train.dataloader import (
    DatasetTextRuleFilter,
    FilterOutRotatedDocuments,
    FrontMatterParser,
    LatexBracketNormalizer,
)


class TestDatasetTextRuleFilter(unittest.TestCase):
    """Test cases for DatasetTextRuleFilter."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = DatasetTextRuleFilter()

    def test_markdown_table_detection(self):
        """Test that markdown tables are correctly detected and filtered."""
        # Sample with markdown table - should be filtered out
        sample_with_md_table = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
Some introduction text.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |
| Data D   | Data E   | Data F   |

Some conclusion text.
                """,
            )
        }
        
        result = self.filter(sample_with_md_table)
        self.assertIsNone(result, "Should filter out samples with markdown tables")

    def test_no_markdown_table(self):
        """Test that samples without markdown tables pass through."""
        sample_without_table = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="This is regular text without any tables. It has | pipes | but not in table format.",
            )
        }
        
        result = self.filter(sample_without_table)
        self.assertIsNotNone(result, "Should pass through samples without markdown tables")
        self.assertEqual(result, sample_without_table)

    def test_valid_html_table(self):
        """Test that valid HTML tables pass through."""
        sample_with_valid_html = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
Some text before table.

<table>
  <tr>
    <td>Cell 1</td>
    <td>Cell 2</td>
  </tr>
  <tr>
    <td>Cell 3</td>
    <td>Cell 4</td>
  </tr>
</table>

Some text after table.
                """,
            )
        }
        
        result = self.filter(sample_with_valid_html)
        self.assertIsNotNone(result, "Should pass through samples with valid HTML tables")

    def test_malformed_html_table_unclosed_tags(self):
        """Test that malformed HTML tables with unclosed tags are filtered out."""
        sample_with_malformed_html = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
Text before.

<table>
  <tr>
    <td>Cell 1</td>
    <td>Cell 2
  <tr>
    <td>Cell 3</td>
    <td>Cell 4</td>
  </tr>

Text after.
                """,
            )
        }
        
        result = self.filter(sample_with_malformed_html)
        self.assertIsNone(result, "Should filter out samples with malformed HTML tables")

    def test_malformed_html_table_missing_closing(self):
        """Test that HTML tables missing closing tags are filtered out."""
        sample_with_unclosed_table = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
Text before.

<table>
  <tr>
    <td>Cell 1</td>
    <td>Cell 2</td>
  </tr>

Text after without closing table tag.
                """,
            )
        }
        
        result = self.filter(sample_with_unclosed_table)
        self.assertIsNone(result, "Should filter out HTML tables without closing tags")

    def test_no_page_data(self):
        """Test that samples without page_data pass through."""
        sample_without_page_data = {
            "markdown_path": Path("/path/to/file.md"),
            "pdf_path": Path("/path/to/file.pdf"),
        }
        
        result = self.filter(sample_without_page_data)
        self.assertIsNotNone(result, "Should pass through samples without page_data")
        self.assertEqual(result, sample_without_page_data)

    def test_no_natural_text(self):
        """Test that samples with page_data but no natural_text pass through."""
        sample_without_text = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text=None,
            )
        }
        
        result = self.filter(sample_without_text)
        self.assertIsNotNone(result, "Should pass through samples without natural_text")

    def test_empty_natural_text(self):
        """Test that samples with empty natural_text pass through."""
        sample_with_empty_text = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="",
            )
        }
        
        result = self.filter(sample_with_empty_text)
        self.assertIsNotNone(result, "Should pass through samples with empty natural_text")

    def test_complex_markdown_table_variations(self):
        """Test various markdown table formats."""
        # Table with alignment indicators
        sample_with_alignment = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
| Left | Center | Right |
|:-----|:------:|------:|
| A    |   B    |     C |
                """,
            )
        }
        
        result = self.filter(sample_with_alignment)
        self.assertIsNone(result, "Should filter out markdown tables with alignment")

    def test_mixed_content(self):
        """Test content with both valid HTML and no markdown tables."""
        sample_mixed = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="""
This document has a valid HTML table:

<table>
  <thead>
    <tr><th>Header 1</th><th>Header 2</th></tr>
  </thead>
  <tbody>
    <tr><td>Data 1</td><td>Data 2</td></tr>
  </tbody>
</table>

But no markdown tables. Just some text with | pipes | that aren't tables.
                """,
            )
        }
        
        result = self.filter(sample_mixed)
        self.assertIsNotNone(result, "Should pass through with valid HTML and no markdown tables")


class TestFilterOutRotatedDocuments(unittest.TestCase):
    """Test cases for FilterOutRotatedDocuments."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = FilterOutRotatedDocuments()

    def test_valid_rotation(self):
        """Test that documents with valid rotation pass through."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="Some text",
            )
        }
        
        result = self.filter(sample)
        self.assertIsNotNone(result, "Should pass through documents with valid rotation")

    def test_invalid_rotation(self):
        """Test that documents with invalid rotation are filtered out."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=False,
                rotation_correction=90,
                is_table=False,
                is_diagram=False,
                natural_text="Some text",
            )
        }
        
        result = self.filter(sample)
        self.assertIsNone(result, "Should filter out documents with invalid rotation")

    def test_rotation_correction_needed(self):
        """Test that documents needing rotation correction are filtered out."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=180,
                is_table=False,
                is_diagram=False,
                natural_text="Some text",
            )
        }
        
        result = self.filter(sample)
        self.assertIsNone(result, "Should filter out documents with non-zero rotation correction")

    def test_no_page_data(self):
        """Test that samples without page_data pass through."""
        sample = {"markdown_path": Path("/path/to/file.md")}
        
        result = self.filter(sample)
        self.assertIsNotNone(result, "Should pass through samples without page_data")


class TestLatexBracketNormalizer(unittest.TestCase):
    """Test cases for LatexBracketNormalizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = LatexBracketNormalizer()

    def test_dollar_to_parentheses(self):
        """Test conversion of $...$ to \\(...\\)."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="The equation $x^2 + y^2 = z^2$ is famous.",
            )
        }
        
        result = self.normalizer(sample)
        expected_text = "The equation \\(x^2 + y^2 = z^2\\) is famous."
        self.assertEqual(result["page_data"].natural_text, expected_text)

    def test_double_dollar_to_brackets(self):
        """Test conversion of $$...$$ to \\[...\\]."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="Display equation:\n$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$",
            )
        }
        
        result = self.normalizer(sample)
        expected_text = "Display equation:\n\\[\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}\\]"
        self.assertEqual(result["page_data"].natural_text, expected_text)

    def test_mixed_latex_delimiters(self):
        """Test handling of mixed inline and display math."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="Inline $a + b$ and display:\n$$c^2 = a^2 + b^2$$\nMore inline $x = y$.",
            )
        }
        
        result = self.normalizer(sample)
        expected_text = "Inline \\(a + b\\) and display:\n\\[c^2 = a^2 + b^2\\]\nMore inline \\(x = y\\)."
        self.assertEqual(result["page_data"].natural_text, expected_text)

    def test_no_latex(self):
        """Test that text without LaTeX passes through unchanged."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text="Regular text without any equations.",
            )
        }
        
        result = self.normalizer(sample)
        self.assertEqual(result["page_data"].natural_text, "Regular text without any equations.")

    def test_no_natural_text(self):
        """Test handling of missing natural_text."""
        sample = {
            "page_data": PageResponse(
                primary_language="en",
                is_rotation_valid=True,
                rotation_correction=0,
                is_table=False,
                is_diagram=False,
                natural_text=None,
            )
        }
        
        result = self.normalizer(sample)
        self.assertIsNone(result["page_data"].natural_text)

    def test_no_page_data(self):
        """Test handling of missing page_data."""
        sample = {"markdown_path": Path("/path/to/file.md")}
        
        result = self.normalizer(sample)
        self.assertEqual(result, sample)


class TestFrontMatterParser(unittest.TestCase):
    """Test cases for FrontMatterParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser_with_class = FrontMatterParser(front_matter_class=PageResponse)
        self.parser_without_class = FrontMatterParser(front_matter_class=None)

    @patch.object(Path, 'read_text')
    def test_parse_yaml_front_matter(self, mock_read_text):
        """Test parsing of YAML front matter."""
        mock_read_text.return_value = """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---
This is the document content.
        """
        
        sample = {"markdown_path": Path("/path/to/file.md")}
        result = self.parser_with_class(sample)
        
        self.assertIn("page_data", result)
        self.assertIsInstance(result["page_data"], PageResponse)
        self.assertEqual(result["page_data"].primary_language, "en")
        self.assertEqual(result["page_data"].natural_text, "This is the document content.")

    @patch.object(Path, 'read_text')
    def test_no_front_matter(self, mock_read_text):
        """Test handling of documents without front matter."""
        mock_read_text.return_value = "Just regular content without front matter."
        
        sample = {"markdown_path": Path("/path/to/file.md")}
        
        # Should raise an error when front_matter_class is specified
        with self.assertRaises(ValueError):
            self.parser_with_class(sample)

    @patch.object(Path, 'read_text')
    def test_malformed_yaml(self, mock_read_text):
        """Test handling of malformed YAML."""
        mock_read_text.return_value = """---
primary_language: en
is_rotation_valid: [this is not valid yaml}
---
Content
        """
        
        sample = {"markdown_path": Path("/path/to/file.md")}
        
        # Parser without class should return empty dict for malformed YAML
        result = self.parser_without_class(sample)
        self.assertEqual(result["page_data"], {})

    @patch.object(Path, 'read_text')
    def test_preserve_existing_markdown_content(self, mock_read_text):
        """Test that existing markdown_content is preserved if present."""
        sample = {
            "markdown_path": Path("/path/to/file.md"),
            "markdown_content": """---
primary_language: fr
is_rotation_valid: true
rotation_correction: 0
is_table: true
is_diagram: false
---
French content."""
        }
        
        # Should not call read_text since markdown_content exists
        result = self.parser_with_class(sample)
        mock_read_text.assert_not_called()
        
        self.assertEqual(result["page_data"].primary_language, "fr")
        self.assertEqual(result["page_data"].is_table, True)


if __name__ == "__main__":
    unittest.main()