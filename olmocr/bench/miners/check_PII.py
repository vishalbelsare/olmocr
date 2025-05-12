#!/usr/bin/env python3

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Dict, List, Tuple

import pydantic
from openai import OpenAI
from tqdm import tqdm

from olmocr.data.renderpdf import render_pdf_to_base64png

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class LanguageCode(str, Enum):
    """Language code enum that can be JSON serialized"""

    en = "English"
    zh = "Chinese"
    hi = "Hindi"
    es = "Spanish"
    fr = "French"
    ar = "Arabic"
    bn = "Bengali"
    ru = "Russian"
    pt = "Portuguese"
    ur = "Urdu"
    id = "Indonesian"
    de = "German"
    ja = "Japanese"
    sw = "Swahili"
    mr = "Marathi"
    te = "Telugu"
    tr = "Turkish"
    vi = "Vietnamese"
    ta = "Tamil"
    ko = "Korean"
    other = "Other"


class PIIAnnotation(pydantic.BaseModel):
    """Structured model for PII annotations returned by ChatGPT"""

    document_description: str
    language_code: LanguageCode
    cannot_read: bool
    inappropriate_content: bool
    is_public_document: bool

    # PII identifiers
    contains_names: bool
    contains_email_addresses: bool
    contains_phone_numbers: bool

    # PII that must co-occur with identifiers
    contains_addresses: bool
    contains_biographical_info: bool  # DOB, gender, etc.
    contains_location_info: bool
    contains_employment_info: bool
    contains_education_info: bool
    contains_medical_info: bool

    # Always sensitive PII
    contains_government_ids: bool
    contains_financial_info: bool
    contains_biometric_data: bool
    contains_login_info: bool

    other_pii: str

    @property
    def has_pii(self) -> bool:
        """Check if the document contains any PII"""
        pii_fields = [
            self.contains_names,
            self.contains_email_addresses,
            self.contains_phone_numbers,
            self.contains_addresses,
            self.contains_biographical_info,
            self.contains_location_info,
            self.contains_employment_info,
            self.contains_education_info,
            self.contains_medical_info,
            self.contains_government_ids,
            self.contains_financial_info,
            self.contains_biometric_data,
            self.contains_login_info,
        ]
        return any(pii_fields) or bool(self.other_pii.strip())

    def get_pii_types(self) -> List[str]:
        """Get a list of all PII types found in the document"""
        pii_types = []

        if self.contains_names:
            pii_types.append("names")
        if self.contains_email_addresses:
            pii_types.append("email")
        if self.contains_phone_numbers:
            pii_types.append("phone")
        if self.contains_addresses:
            pii_types.append("addresses")
        if self.contains_biographical_info:
            pii_types.append("biographical")
        if self.contains_location_info:
            pii_types.append("location")
        if self.contains_employment_info:
            pii_types.append("employment")
        if self.contains_education_info:
            pii_types.append("education")
        if self.contains_medical_info:
            pii_types.append("medical")
        if self.contains_government_ids:
            pii_types.append("government-id")
        if self.contains_financial_info:
            pii_types.append("financial")
        if self.contains_biometric_data:
            pii_types.append("biometric")
        if self.contains_login_info:
            pii_types.append("login-info")
        if self.other_pii.strip():
            pii_types.append("other")

        return pii_types

    def dict(self):
        """Convert the model to a dictionary with serializable values"""
        result = super().model_dump()
        # Convert LanguageCode enum to string
        if isinstance(result["language_code"], LanguageCode):
            result["language_code"] = result["language_code"].value
        return result


def analyze_pdf_page(pdf_path, page_num, openai_model="gpt-4.1"):
    """Analyze a PDF page for PII using OpenAI vision model."""
    try:
        # Check if the file exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file {pdf_path} does not exist")
            return None

        # Render PDF to base64 image
        base64_image = render_pdf_to_base64png(pdf_path, page_num)
        if not base64_image:
            print(f"Error: Could not render PDF {pdf_path} page {page_num}")
            return None

        # Prepare the user message with all instructions
        user_message = """
You are a document analyzer that identifies Personally Identifiable Information (PII) in documents. 
Your task is to analyze the provided document image and determine:
1. Whether the document is intended for public release or dissemination (e.g., research paper, public report, etc.)
2. If the document contains any PII

For PII identification, follow these specific guidelines:

IDENTIFIERS FOR PII:
The following are considered identifiers that can make information PII:
- Names (full names, first names, last names, nicknames)
- Email addresses
- Phone numbers

PII THAT MUST CO-OCCUR WITH AN IDENTIFIER:
The following types of information should ONLY be marked as PII if they occur ALONGSIDE an identifier (commonly, a person's name):
- Addresses (street address, postal code, etc.)
- Biographical Information (date of birth, place of birth, gender, sexual orientation, race, ethnicity, citizenship/immigration status, religion)
- Location Information (geolocations, specific coordinates)
- Employment Information (job titles, workplace names, employment history)
- Education Information (school names, degrees, transcripts)
- Medical Information (health records, diagnoses, genetic or neural data)

PII THAT OCCURS EVEN WITHOUT AN IDENTIFIER:
The following should ALWAYS be marked as PII even if they do not occur alongside an identifier:
- Government IDs (Social Security Numbers, passport numbers, driver's license numbers, tax IDs)
- Financial Information (credit card numbers, bank account/routing numbers)
- Biometric Data (fingerprints, retina scans, facial recognition data, voice signatures)
- Login information (ONLY mark as PII when a username, password, and login location are present together)

If the document is a form, then only consider fields which are filled out with specific values as potential PII.
If this page does not itself contain PII, but references documents (such as curriculum vitae, personal statements) that typically contain PII, then do not mark it as PII.
Only consider actual occurrences of the PII within the document shown.
"""

        # Use the chat completions API with the custom schema
        completion = client.beta.chat.completions.parse(
            model=openai_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}],
                }
            ],
            response_format=PIIAnnotation,
            max_tokens=1000,
        )

        return completion.choices[0].message.parsed

    except Exception as e:
        print(f"Error analyzing PDF {pdf_path} (page {page_num}): {e}")
        return None


def process_pdf(pdf_path, page_num, original_data_entries, pdf_base_dir, openai_model):
    """Process a PDF and apply the result to all entries with the same PDF."""
    try:
        # Build the full path to the PDF
        full_pdf_path = os.path.join(pdf_base_dir, pdf_path)

        if os.path.exists(full_pdf_path):
            # Analyze the PDF page
            print(f"Analyzing {full_pdf_path} page {page_num}")
            pii_annotation = analyze_pdf_page(full_pdf_path, page_num, openai_model)

            if pii_annotation:
                results = []
                # Apply the annotation to all entries with this PDF path and page
                for data in original_data_entries:
                    # Create a new copy of the data to avoid modifying the original
                    data_copy = data.copy()
                    # Add the PII annotation to the data
                    data_copy["pii_annotation"] = pii_annotation.dict()

                    # Check if this entry meets our criteria: has_pii and not is_public_document
                    if pii_annotation.has_pii and not pii_annotation.is_public_document:
                        results.append(data_copy)

                return results
        else:
            print(f"Warning: PDF file {full_pdf_path} not found")

        return []
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Analyze PDFs for PII")
    parser.add_argument("input_file", help="Input JSONL file path with PDF references")
    parser.add_argument("output_file", help="Output JSONL file path for entries with PII")
    parser.add_argument("--pdf-dir", default="olmOCR-bench/bench_data/pdfs", help="Base directory for PDF files")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI vision model to use")
    parser.add_argument("--max-workers", type=int, default=32, help="Maximum number of worker threads")
    parser.add_argument("--limit", type=int, help="Limit the number of entries to process (for testing)")
    args = parser.parse_args()

    # Make sure OpenAI API key is set
    if not client.api_key:
        print("ERROR: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return

    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file {args.input_file} does not exist.")
        return

    if not os.path.exists(args.pdf_dir):
        print(f"ERROR: PDF directory {args.pdf_dir} does not exist.")
        return

    # Clear the output file
    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    # Read all entries from the input file
    entries = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {line}")

    # Apply limit if specified
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    print(f"Processing {len(entries)} entries from {args.input_file}")

    # Group entries by unique PDF and page
    pdf_page_entries: Dict[Tuple[str, int], List[Dict]] = {}

    for entry in entries:
        pdf_path = entry.get("pdf", "")
        page_num = entry.get("page", 1)

        # Use a tuple of (pdf_path, page_num) as the key
        key = (pdf_path, page_num)

        if key not in pdf_page_entries:
            pdf_page_entries[key] = []

        pdf_page_entries[key].append(entry)

    print(f"Found {len(pdf_page_entries)} unique PDF pages to analyze")

    # Process unique PDFs in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a list of futures
        futures = []
        for (pdf_path, page_num), data_entries in pdf_page_entries.items():
            future = executor.submit(process_pdf, pdf_path, page_num, data_entries, args.pdf_dir, args.model)
            futures.append(future)

        # Process results as they complete with tqdm progress bar
        for future in tqdm(futures, total=len(futures), desc="Analyzing unique PDFs for PII"):
            results = future.result()
            if results:  # Only append non-empty results
                all_results.extend(results)
                # Write incrementally to output file to avoid losing progress
                with open(args.output_file, "a") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                print(f"Found {len(results)} entries with PII for {results[0].get('pdf', '')} page {results[0].get('page', '')}")

    # Make sure the final count is accurate
    print(f"Found {len(all_results)} total entries with PII (has PII and not public document). Results written to {args.output_file}")


if __name__ == "__main__":
    main()
