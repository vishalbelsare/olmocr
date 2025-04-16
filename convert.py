# import os
# from PIL import Image
# import argparse

# def convert_images_to_pdfs(input_folder, output_folder):
#     # Create the output folder if it doesn't exist.
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Supported image extensions.
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
#     # Loop through all files in the input folder.
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(image_extensions):
#             input_path = os.path.join(input_folder, filename)
#             # Open the image file.
#             try:
#                 with Image.open(input_path) as img:
#                     # Convert image to RGB (required for PDF conversion)
#                     rgb_img = img.convert("RGB")
#                     # Define the output PDF filename.
#                     base_name, _ = os.path.splitext(filename)
#                     output_pdf = os.path.join(output_folder, base_name + ".pdf")
#                     # Save the image as PDF.
#                     rgb_img.save(output_pdf, "PDF")
#                     print(f"Converted {filename} -> {output_pdf}")
#             except Exception as e:
#                 print(f"Failed to convert {filename}: {e}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert all images in a folder to PDFs.")
#     parser.add_argument("input_folder", help="Folder containing image files.")
#     parser.add_argument("output_folder", help="Folder to save converted PDFs.")
#     args = parser.parse_args()
    
#     convert_images_to_pdfs(args.input_folder, args.output_folder)


import os
import random
from PyPDF2 import PdfReader, PdfWriter

# Path to your original PDF
input_pdf_path = 'long_texts/pdfs/9.pdf'

# Load the PDF
reader = PdfReader(input_pdf_path)
total_pages = len(reader.pages)

# How many pages to extract
num_pages_to_extract = 6

# Random unique page numbers (0-based for code, +1 when naming)
random_page_numbers = sorted(random.sample(range(total_pages), num_pages_to_extract))
print(f"Randomly selected pages: {[num + 1 for num in random_page_numbers]}")

# Output directory (same as input file)
output_dir = os.path.dirname(input_pdf_path)

# Extract and save each page individually
for page_num in random_page_numbers:
    writer = PdfWriter()
    writer.add_page(reader.pages[page_num])

    output_filename = os.path.join(output_dir, f"9_pg{page_num + 1}.pdf")
    with open(output_filename, 'wb') as out_file:
        writer.write(out_file)
    
    print(f"Saved: {output_filename}")