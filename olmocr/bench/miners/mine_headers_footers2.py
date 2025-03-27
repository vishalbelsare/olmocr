#!/usr/bin/env python3
"""
mine_headers_footers.py - Extract headers and footers from PDF documents.

This script:
1. Takes a directory containing PDF documents as input
2. For each PDF, extracts a random page and renders it to an image
3. Uses Gemini to identify headers and footers in the rendered image
4. Creates a test file asserting that the header/footer text should not appear
5. Extracts the page from the PDF and saves it to an output folder

Usage:
  python mine_headers_footers.py --input_dir path/to/pdfs --output_dir path/to/output --api_key your_gemini_api_key
"""

import argparse
import base64
import json
import os
import random
from typing import List, Optional

import pypdf
from google import genai
from google.genai import types
from tqdm import tqdm

from olmocr.bench.tests import TextPresenceTest, save_tests
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.filter import PdfFilter


def extract_page_from_pdf(input_path: str, output_path: str, page_num: int) -> bool:
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (0-indexed)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the input PDF
        reader = pypdf.PdfReader(input_path)

        # Check if page number is valid
        if page_num >= len(reader.pages):
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[page_num])

        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        raise


def detect_headers_footers(pdf_path: str, page_num: int, api_key: str) -> Optional[List[str]]:
    """
    Use Gemini to detect headers and footers in a rendered PDF page.

    Args:
        pdf_path: Path to the PDF file
        page_num: The page number to analyze (0-indexed)
        api_key: Gemini API key

    Returns:
        Optional[List[str]]: List of detected header/footer texts, or None if detection failed
    """
    client = genai.Client(
        api_key=api_key,  # Use the provided API key
    )
    model = "gemini-2.0-flash"

    # Render the PDF page as an image
    try:
        image_base64 = render_pdf_to_base64png(pdf_path, page_num=page_num + 1, target_longest_image_dim=2048)  # render_pdf_to_base64png is 1-indexed
    except Exception as e:
        print(f"Error rendering PDF page: {str(e)}")
        return None

    image_part = types.Part(inline_data=types.Blob(mime_type="image/png", data=base64.b64decode(image_base64)))

    contents = [
        types.Content(
            role="user",
            parts=[
                image_part,
                types.Part.from_text(
                    text="""Please extract and display the complete text from the document without omission. Include all sections and ensure nothing is summarized or abbreviated. I want the entire text of the document at any cost. Do not hallucinate."""
                    # text="""Please tell me which text in this image is part of any headers/footers and would therefore be skipped it someone were reading it outloud to another person. Include page numbers and document-level headers and footers, but not inner subsections."""
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "headers": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
                "footers": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                ),
            },
        ),
    )

    response = client.models.generate_content(model=model, contents=contents, config=generate_content_config)

    assert len(response.candidates) > 0, "No candidates found"
    assert response.candidates[0].finish_reason == types.FinishReason.STOP, "Finish reason was not STOP, likely a processing error or repetition failure"

    data = json.loads(response.candidates[0].content.parts[0].text)

    return data.get("headers", []) + data.get("footers", [])


def process_pdf(pdf_path: str, output_dir: str, api_key: str, tests: List[TextPresenceTest]) -> None:
    """
    Process a single local PDF file.

    Args:
        pdf_path: Path to the local PDF file
        output_dir: Directory for output files
        api_key: Gemini API key
        tests: List to append tests to
    """
    # Extract filename from path
    pdf_filename = os.path.basename(pdf_path)

    pdf_filter = PdfFilter()

    if pdf_filter.filter_out_pdf(pdf_path):
        print("Filtering out", pdf_filename)
        return

    try:
        # Read the PDF to get the number of pages
        reader = pypdf.PdfReader(pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF {pdf_filename} has no pages")
            return

        all_pages = list(range(len(reader.pages)))
        random.shuffle(all_pages)

        for page_num in all_pages:
            # Detect headers and footers
            header_footer_text = detect_headers_footers(pdf_path, page_num, api_key)

            # Only stick with headers and footers that have some actual data in them
            if header_footer_text:
                header_footer_text = [x for x in header_footer_text if len(x.strip()) > 3]

            if not header_footer_text:
                print(f"No headers/footers detected in {pdf_filename} page {page_num}")
                continue

            # Extract the page and save to output dir
            pdf_basename = os.path.splitext(pdf_filename)[0]
            output_pdf_path = os.path.join(output_dir, "pdfs", f"{pdf_basename}_pg{page_num+1}.pdf")

            extract_page_from_pdf(pdf_path, output_pdf_path, page_num)

            # Create tests for each header/footer text
            for i, text in enumerate(header_footer_text):
                test_id = f"{pdf_basename}_pg{page_num+1}_header_{i:02d}"
                test = TextPresenceTest(
                    id=test_id,
                    pdf=f"{pdf_basename}_pg{page_num+1}.pdf",
                    page=1,  # The extracted PDF has only one page
                    type="absent",
                    text=text,
                    max_diffs=0,
                )
                tests.append(test)

            print(f"Processed {pdf_filename} page {page_num+1}, found {len(header_footer_text)} headers/footers")
            return  # Process only one page per PDF (as in the original code)

    except Exception as e:
        print(f"Error processing {pdf_filename}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Extract headers and footers from PDF documents")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--api_key", help="Gemini API key (if not provided, will use GEMINI_API_KEY environment variable)")
    parser.add_argument("--limit", type=int, default=500, help="Maximum number of tests to generate (default: 500)")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Gemini API key not provided. Use --api_key or set GEMINI_API_KEY environment variable.")
        return

    # Create output directory
    os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)

    # Get all PDF files from the input directory
    pdf_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    print(f"Found {len(pdf_files)} PDF files in input directory")

    # Process each PDF
    tests = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        process_pdf(pdf_path, args.output_dir, api_key, tests)

        # Save tests after each PDF to avoid losing data in case of crashes
        if tests:
            save_tests(tests, os.path.join(args.output_dir, "header_footer_tests.jsonl"))

        if len(tests) >= args.limit:
            print(f"Reached limit of {args.limit} tests. Stopping.")
            break

    print(f"Saved {len(tests)} tests to {os.path.join(args.output_dir, 'header_footer_tests.jsonl')}")


if __name__ == "__main__":
    main()