import os
from PyPDF2 import PdfReader, PdfWriter, errors

def split_pdf(pdf_path):
    """
    Splits a multi-page PDF into individual pages.
    If the PDF has only one page, it is left as-is.
    If the PDF cannot be processed (e.g., encryption issues), it is skipped.
    After splitting a multi-page PDF, the original file is removed.
    """
    base_dir = os.path.dirname(pdf_path)
    base_name = os.path.basename(pdf_path)
    name, ext = os.path.splitext(base_name)

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except errors.DependencyError as de:
        print(f"Error processing {base_name}: {de}")
        print("Please install PyCryptodome using: pip install pycryptodome")
        return
    except Exception as e:
        print(f"Failed to open {base_name}: {e}")
        return

    # If only one page, leave the file as is.
    if num_pages <= 1:
        print(f"{base_name} has only one page. Leaving it as is.")
        return

    # For each page, create a new PDF file
    for i in range(num_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        output_filename = f"{name}_page_{i+1}.pdf"
        output_path = os.path.join(base_dir, output_filename)
        try:
            with open(output_path, "wb") as out_file:
                writer.write(out_file)
            print(f"Created {output_filename}")
        except Exception as e:
            print(f"Failed to write {output_filename}: {e}")

    # After splitting, remove the original multi-page PDF
    try:
        os.remove(pdf_path)
        print(f"Removed original file {base_name}")
    except Exception as e:
        print(f"Failed to remove original file {base_name}: {e}")

def split_all_pdfs_in_folder(folder_path):
    """
    Processes all PDF files in the given folder.
    """
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")
            split_pdf(pdf_path)

if __name__ == "__main__":
    folder = input("Enter the folder path containing PDFs: ").strip()
    split_all_pdfs_in_folder(folder)