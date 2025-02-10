import os
import json
import tempfile
import boto3
import concurrent.futures
from tqdm import tqdm

from olmocr.s3_utils import get_s3_bytes, expand_s3_glob
from olmocr.data.buildsilver import build_page_query

# Initialize the boto3 S3 client.
s3 = boto3.client("s3")
# Expand the S3 glob to get a list of all PDF S3 URIs.
all_pdfs = expand_s3_glob(s3, "s3://ai2-oe-data/jakep/pdfdata/pdelfin_testset/*.pdf")

print(f"Found {len(all_pdfs)} PDFs.")

def process_pdf(pdf_s3_uri):
    """
    Downloads a PDF from S3, writes it to a temporary file,
    builds a query dictionary from the PDF, and cleans up the temporary file.
    Returns the natural_text string.
    """
    local_pdf_path = None
    try:
        # Download the PDF as bytes from S3.
        pdf_bytes = get_s3_bytes(s3, pdf_s3_uri)
        
        # Write the PDF bytes to a temporary file so that build_page_query can work on a file path.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            local_pdf_path = tmp_file.name

        # Build the query dictionary from the PDF.
        example = build_page_query(local_pdf_path, pdf_s3_uri, 1)
        print(example.choices[0].message.content)

        data = json.loads(example.choices[0].message.content)
        return data["natural_text"]
       
    except Exception as e:
        print(f"Error processing {pdf_s3_uri}: {e}")
        return None

    finally:
        # Remove the temporary file if it exists.
        if local_pdf_path and os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)

# Use ThreadPoolExecutor to process PDFs concurrently.
max_workers = 4  # Adjust the number of threads as needed.
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all the PDF processing tasks.
    future_to_pdf = {executor.submit(process_pdf, pdf_uri): pdf_uri for pdf_uri in all_pdfs}

    # Use tqdm to display a progress bar as tasks complete.
    for future in tqdm(concurrent.futures.as_completed(future_to_pdf),
                       total=len(future_to_pdf), desc="Processing PDFs"):
        pdf_s3_uri = future_to_pdf[future]
        result = future.result()
        if result is not None:
            # Construct the output filename.
            # Here we take the base name (e.g., "document.pdf") and replace ".pdf" with "_gpt4o.md".
            pdf_basename = os.path.basename(pdf_s3_uri)
            output_filename = pdf_basename.replace(".pdf", "_gpt4o.md")
            # Optionally, set an output directory (this example uses "output").
            output_dir = "gpt4otestset_output"
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, output_filename)
            with open(output_filepath, "w", encoding="utf-8") as outfile:
                outfile.write(result)
            print(f"Wrote result to {output_filepath}")

print("Processing complete.")
