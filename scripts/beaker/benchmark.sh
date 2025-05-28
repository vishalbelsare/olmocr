# download olmocr-bench dataset
apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/allenai/olmOCR-bench
cd olmOCR-bench/
git lfs pull
cd ..

python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/arxiv_math/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/headers_footers/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/long_tiny_text/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/multi_column/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/old_scans/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/old_scans_math/*.pdf
python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/tables/*.pdf

python olmocr/bench/workspace_to_benchmark.py localworkspace/ olmOCR-bench/bench_data/markdown_output --bench-path ./olmOCR-bench/
python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data