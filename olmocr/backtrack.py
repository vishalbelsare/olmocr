import os
import re
import boto3
import pypdf
import argparse
import concurrent.futures
from typing import List, Tuple, Dict
from botocore.exceptions import ClientError


def download_pdf_from_s3(s3_path: str, local_path: str) -> bool:
    """
    Download a PDF file from S3.
    Args:
        s3_path: The S3 path (s3://bucket/path/to/file.pdf)
        local_path: The local path to save the file
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Parse S3 path
        parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        
        # Create S3 client
        s3 = boto3.client("s3")
        
        # Check if the file exists before downloading
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"  File not found in S3: {s3_path}")
            else:
                print(f"  Error checking file existence: {str(e)}")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"  Error downloading {s3_path}: {str(e)}")
        return False


def extract_page_from_pdf(input_path: str, output_path: str, page_num: int) -> bool:
    """
    Extract a specific page from a PDF and save it as a new PDF.
    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (1-indexed)
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read the input PDF
        reader = pypdf.PdfReader(input_path)
        
        # Convert from 1-indexed to 0-indexed for pypdf
        zero_indexed_page = page_num - 1
        
        # Check if page number is valid
        if zero_indexed_page >= len(reader.pages) or zero_indexed_page < 0:
            print(f"  Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False
            
        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[zero_indexed_page])
        
        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        return True
    except Exception as e:
        print(f"  Error extracting page {page_num} from {input_path}: {str(e)}")
        return False


def extract_file_info_and_page(path: str) -> Tuple[str, str, int]:
    """
    Extract the folder name, file ID, and page number from a processed path.
    
    Args:
        path: Processed S3 path like 'path/edafXXXXXXX_page_X_processed.pdf'
    Returns:
        Tuple of (folder_name, file_id, page_number)
    """
    # Extract the filename from the path
    filename = os.path.basename(path)
    
    # Check for the processed file pattern
    match = re.search(r'(.+?)_page_(\d+)_processed\.pdf$', filename)
    if match:
        full_id = match.group(1)
        page_num = int(match.group(2))
        
        # The first 4 characters are the folder name
        if len(full_id) >= 4:
            folder_name = full_id[:4]
            file_id = full_id[4:]  # Rest is the file ID
            return folder_name, file_id, page_num
    
    # If it's just a regular PDF, try to parse it similarly
    if filename.endswith('.pdf'):
        full_id = filename.replace('.pdf', '')
        if len(full_id) >= 4:
            folder_name = full_id[:4]
            file_id = full_id[4:]
            return folder_name, file_id, 1
    
    raise ValueError(f"Invalid file path format or ID too short: {path}")


def get_actual_s3_path(folder_name: str, file_id: str) -> str:
    """
    Get the actual S3 path using the folder-based mapping pattern.
    
    Args:
        folder_name: The four-letter folder name (first 4 chars of original ID)
        file_id: The file ID (rest of the original ID)
    Returns:
        The actual S3 path in the format s3://ai2-s2-pdfs/{folder_name}/{file_id}.pdf
    """
    return f"s3://ai2-s2-pdfs/{folder_name}/{file_id}.pdf"


def process_single_file(processed_path: str, output_dir: str) -> Dict:
    """
    Process a single file path, download the original, and extract pages.
    This function is designed to be used with concurrent.futures.
    
    Args:
        processed_path: S3 path to process
        output_dir: Directory to save downloaded and processed files
    
    Returns:
        Dict with processing results for this file
    """
    result = {
        'path': processed_path,
        'status': 'parse_failed',
        'output_path': None
    }
    
    print(f"Processing: {processed_path}")
    
    try:
        # Extract folder name, file ID and page number
        try:
            folder_name, file_id, page_num = extract_file_info_and_page(processed_path)
        except ValueError as e:
            print(f"  {str(e)}")
            return result
        
        # Get the actual S3 path using folder-based mapping
        actual_s3_path = get_actual_s3_path(folder_name, file_id)
            
        # Local paths
        # Create a structure that mirrors the output path format
        output_dir_struct = os.path.join(output_dir, os.path.dirname(processed_path.replace("s3://", "")))
        original_basename = f"{folder_name}{file_id}.pdf"
        processed_basename = f"{folder_name}{file_id}_page_{page_num}_processed.pdf"
        
        original_local_path = os.path.join(output_dir_struct, original_basename)
        processed_local_path = os.path.join(output_dir_struct, processed_basename)
        
        # Download the original PDF
        if download_pdf_from_s3(actual_s3_path, original_local_path):
            # Extract the specified page
            if extract_page_from_pdf(original_local_path, processed_local_path, page_num):
                result['status'] = 'success'
                result['output_path'] = processed_local_path
                print(f"  Successfully processed: {processed_local_path}")
            else:
                result['status'] = 'extraction_failed'
                print(f"  Failed to extract page {page_num}")
        else:
            result['status'] = 'download_failed'
            print(f"  Failed to download {actual_s3_path}")
    except Exception as e:
        print(f"  Error processing {processed_path}: {str(e)}")
    
    return result


def process_file_list(file_list_path: str, output_dir: str, max_workers: int = None) -> Dict[str, List[str]]:
    """
    Process a list of file paths in parallel using ThreadPoolExecutor.
    
    Args:
        file_list_path: Path to a text file containing S3 paths
        output_dir: Directory to save downloaded and processed files
        max_workers: Maximum number of worker threads (None = auto-determine)
    
    Returns:
        Dict containing lists of successful and failed paths
    """
    results = {
        'success': [],
        'download_failed': [],
        'extraction_failed': [],
        'parse_failed': []
    }
    
    # Read the list of file paths
    with open(file_list_path, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    
    total = len(paths)
    print(f"Processing {total} files with {max_workers if max_workers else 'auto'} workers")
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_file, path, output_dir): path
            for path in paths
        }
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_path)):
            path = future_to_path[future]
            try:
                result = future.result()
                status = result['status']
                results[status].append(path)
                
                # Periodic progress update
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    print(f"Progress: {i+1}/{total} files processed")
                    
            except Exception as e:
                print(f"Unexpected error processing {path}: {str(e)}")
                results['parse_failed'].append(path)
    
    return results


def write_report(results: Dict[str, List[str]], output_dir: str) -> None:
    """
    Write a report of the processing results.
    Args:
        results: Dictionary of results
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, "processing_report.txt")
    
    with open(report_path, "w") as f:
        f.write("PDF Processing Report\n")
        f.write("===================\n\n")
        
        f.write(f"Total files processed: {sum(len(v) for v in results.values())}\n")
        f.write(f"Successfully processed: {len(results['success'])}\n")
        f.write(f"Failed downloads: {len(results['download_failed'])}\n")
        f.write(f"Failed extractions: {len(results['extraction_failed'])}\n")
        f.write(f"Failed parsing: {len(results['parse_failed'])}\n\n")
        
        if results['download_failed']:
            f.write("Failed Downloads:\n")
            f.write("----------------\n")
            for path in results['download_failed']:
                f.write(f"  {path}\n")
            f.write("\n")
            
        if results['extraction_failed']:
            f.write("Failed Extractions:\n")
            f.write("------------------\n")
            for path in results['extraction_failed']:
                f.write(f"  {path}\n")
            f.write("\n")
            
        if results['parse_failed']:
            f.write("Failed Parsing:\n")
            f.write("--------------\n")
            for path in results['parse_failed']:
                f.write(f"  {path}\n")
    
    print(f"Report written to {report_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download PDFs from S3 and extract specific pages")
    parser.add_argument("file_list", help="Path to a text file containing S3 paths")
    parser.add_argument("--output-dir", default="./output", help="Directory to save downloaded and processed files")
    parser.add_argument("--max-workers", type=int, default=None, 
                        help="Maximum number of worker threads (default: auto-determine based on CPU count)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing if errors occur")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the file list
    results = process_file_list(args.file_list, args.output_dir, args.max_workers)
    
    # Write report
    write_report(results, args.output_dir)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"  Successfully processed: {len(results['success'])}")
    print(f"  Failed downloads: {len(results['download_failed'])}")
    print(f"  Failed extractions: {len(results['extraction_failed'])}")
    print(f"  Failed parsing: {len(results['parse_failed'])}")
    print(f"\nSee {os.path.join(args.output_dir, 'processing_report.txt')} for details")


if __name__ == "__main__":
    main()