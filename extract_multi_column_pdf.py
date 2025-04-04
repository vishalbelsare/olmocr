#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

app = Flask(__name__)

# Global state
DATASET_DIR = ""
CURRENT_PDF = None
ALL_PDFS = []
MULTIPLE_COLUMNS_DIR = None

def create_output_dir():
    """Create multiple columns directory if it doesn't exist."""
    os.makedirs(MULTIPLE_COLUMNS_DIR, exist_ok=True)

def find_next_pdf() -> Optional[str]:
    """Find the next PDF in the list."""
    global ALL_PDFS, CURRENT_PDF
    
    if not CURRENT_PDF or CURRENT_PDF not in ALL_PDFS:
        if ALL_PDFS:
            return ALL_PDFS[0]
        return None
    
    current_index = ALL_PDFS.index(CURRENT_PDF)
    if current_index < len(ALL_PDFS) - 1:
        return ALL_PDFS[current_index + 1]
    return None

@app.route("/pdf/<path:pdf_name>")
def serve_pdf(pdf_name):
    """Serve the PDF file directly."""
    pdf_path = os.path.join(DATASET_DIR, pdf_name)
    return send_file(pdf_path, mimetype="application/pdf")

@app.route("/")
def index():
    """Main page displaying the current PDF and controls."""
    global CURRENT_PDF, ALL_PDFS

    # If no current PDF is set, find the next one
    if CURRENT_PDF is None:
        CURRENT_PDF = find_next_pdf()

    # If still no PDF, all PDFs have been processed
    if CURRENT_PDF is None:
        return render_template("all_done.html")

    # Create PDF URL for the viewer
    pdf_url = url_for("serve_pdf", pdf_name=CURRENT_PDF)

    # Calculate progress
    current_index = ALL_PDFS.index(CURRENT_PDF) if CURRENT_PDF in ALL_PDFS else 0
    progress_percent = int((current_index + 1) / len(ALL_PDFS) * 100) if ALL_PDFS else 0

    return render_template(
        "review.html",
        pdf_name=CURRENT_PDF,
        pdf_path=pdf_url,
        pdf_index=current_index,
        total_pdfs=len(ALL_PDFS),
        progress_percent=progress_percent,
    )

@app.route("/next_pdf", methods=["POST"])
def next_pdf():
    """Move to the next PDF in the list."""
    global CURRENT_PDF, ALL_PDFS

    if CURRENT_PDF in ALL_PDFS:
        current_index = ALL_PDFS.index(CURRENT_PDF)
        if current_index < len(ALL_PDFS) - 1:
            CURRENT_PDF = ALL_PDFS[current_index + 1]
        else:
            CURRENT_PDF = None  # No more PDFs

    return redirect(url_for("index"))

@app.route("/prev_pdf", methods=["POST"])
def prev_pdf():
    """Move to the previous PDF in the list."""
    global CURRENT_PDF, ALL_PDFS

    if CURRENT_PDF in ALL_PDFS:
        current_index = ALL_PDFS.index(CURRENT_PDF)
        if current_index > 0:
            CURRENT_PDF = ALL_PDFS[current_index - 1]

    return redirect(url_for("index"))

@app.route("/yes_button", methods=["POST"])
def yes_button():
    """Handle Yes button - move PDF to multiple_columns folder."""
    global CURRENT_PDF, ALL_PDFS, DATASET_DIR, MULTIPLE_COLUMNS_DIR
    
    if CURRENT_PDF and CURRENT_PDF in ALL_PDFS:
        # Move the PDF file
        source_path = os.path.join(DATASET_DIR, CURRENT_PDF)
        dest_path = os.path.join(MULTIPLE_COLUMNS_DIR, CURRENT_PDF)
        
        try:
            shutil.move(source_path, dest_path)
            print(f"Moved {CURRENT_PDF} to multiple_columns folder")
        except Exception as e:
            print(f"Error moving file: {str(e)}")
        
        # Move to next PDF
        next_pdf = find_next_pdf()
        CURRENT_PDF = next_pdf
    
    return redirect(url_for("index"))

@app.route("/no_button", methods=["POST"])
def no_button():
    """Handle No button - keep PDF in original folder and move to next."""
    global CURRENT_PDF
    
    # Just move to the next PDF
    next_pdf = find_next_pdf()
    CURRENT_PDF = next_pdf
    
    return redirect(url_for("index"))

def create_templates_directory():
    """Create templates directory for Flask."""
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create review.html template
    review_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Classifier: {{ pdf_name }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .header {
                padding: 10px 20px;
                background-color: #333;
                color: white;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .content {
                display: flex;
                flex: 1;
                overflow: hidden;
            }
            .pdf-viewer {
                flex: 1;
                height: 100%;
                border: none;
            }
            .sidebar {
                width: 250px;
                background-color: #f5f5f5;
                padding: 20px;
                border-left: 1px solid #ddd;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .button {
                padding: 12px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                text-align: center;
                width: 100%;
            }
            .yes-button {
                background-color: #4CAF50;
                color: white;
            }
            .no-button {
                background-color: #f44336;
                color: white;
            }
            .nav-button {
                background-color: #555;
                color: white;
            }
            .progress-container {
                width: 100%;
                background-color: #ddd;
                border-radius: 4px;
            }
            .progress-bar {
                height: 20px;
                background-color: #4CAF50;
                border-radius: 4px;
                text-align: center;
                color: white;
                line-height: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>PDF Classifier: {{ pdf_name }}</h2>
            <span>PDF {{ pdf_index + 1 }} of {{ total_pdfs }}</span>
        </div>
        <div class="content">
            <iframe class="pdf-viewer" src="{{ pdf_path }}"></iframe>
            <div class="sidebar">
                <h3>Is this a multiple-column PDF?</h3>
                <form action="/yes_button" method="post">
                    <button class="button yes-button" type="submit">YES - Move to multiple_columns</button>
                </form>
                <form action="/no_button" method="post">
                    <button class="button no-button" type="submit">NO - Keep in original folder</button>
                </form>
                
                <h3>Navigation</h3>
                <form action="/prev_pdf" method="post">
                    <button class="button nav-button" type="submit">Previous PDF</button>
                </form>
                <form action="/next_pdf" method="post">
                    <button class="button nav-button" type="submit">Skip PDF</button>
                </form>
                
                <h3>Progress</h3>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {{ progress_percent }}%">{{ progress_percent }}%</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create all_done.html template
    all_done_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>All PDFs Processed</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f5f5f5;
            }
            .message {
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            h1 {
                color: #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="message">
            <h1>All PDFs Processed!</h1>
            <p>You have completed reviewing all PDF files.</p>
            <p>Multiple-column PDFs have been moved to the designated folder.</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(templates_dir, "review.html"), "w") as f:
        f.write(review_html)
        
    with open(os.path.join(templates_dir, "all_done.html"), "w") as f:
        f.write(all_done_html)

def load_pdfs(directory):
    """Load all PDF files from the directory."""
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_files.append(filename)
    return sorted(pdf_files)

def main():
    """Main entry point with command-line arguments."""
    global DATASET_DIR, ALL_PDFS, MULTIPLE_COLUMNS_DIR

    parser = argparse.ArgumentParser(description="PDF Column Classifier")
    parser.add_argument("pdf_dir", help="Directory containing PDF files")
    parser.add_argument("--output", help="Output directory for multiple-column PDFs", default=None)
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Flask app")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")

    args = parser.parse_args()

    # Validate PDF directory
    if not os.path.isdir(args.pdf_dir):
        print(f"Error: PDF directory not found: {args.pdf_dir}")
        return 1

    # Set global variables
    DATASET_DIR = os.path.abspath(args.pdf_dir)
    if args.output:
        MULTIPLE_COLUMNS_DIR = os.path.abspath(args.output)
    else:
        MULTIPLE_COLUMNS_DIR = os.path.join(DATASET_DIR, "multiple_columns")
    
    # Create output directory
    create_output_dir()

    # Load PDFs
    ALL_PDFS = load_pdfs(DATASET_DIR)
    if not ALL_PDFS:
        print("No PDF files found in the directory.")
        return 1

    # Create templates
    create_templates_directory()

    # Start Flask app with reloader disabled
    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Found {len(ALL_PDFS)} PDF files to classify")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

    return 0

if __name__ == "__main__":
    sys.exit(main())