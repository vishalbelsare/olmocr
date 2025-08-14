# olmOCR Training Guide

This guide provides comprehensive instructions for training olmOCR models, including what you need to reproduce https://huggingface.co/allenai/olmOCR-7B-0725-FP8 on your own hardware.

## Environment setup

The first step is to setup your python/conda environment, and set things up the same way as for running olmocr.

Then, add in some extra training requirements:

```bash
pip install .[train]
pip install transformers==4.52.4
pip install flash-attn==2.8.0.post2 --no-build-isolation
```


### Dataset Format

The training data should be organized as pairs of PDF files and their corresponding markdown annotations:

**Important: Each PDF needs to be a single page only!** 

```
data/
├── document1.pdf
├── document1.md
├── document2.pdf
├── document2.md
└── ...
```

Each markdown file should contain:
1. YAML front matter with metadata
2. The extracted text content

Example markdown format:
```markdown
---
primary_language: en
is_rotation_valid: True
rotation_correction: 0
is_table: False
is_diagram: False
---
Document text goes here...
```

The easiest way to grab a lot of files in this format is to use `prepare_olmocrmix.py` which will automatically download and prepare 
[olmOCR-mix-0225](https://huggingface.co/datasets/allenai/olmOCR-mix-0225) for your environment.

```bash
# Caution, requires ~200GB of disk space
python olmocr/train/prepare_olmocrmix.py --subset 01_books --split eval_iabooks --destination ~/olmOCR-mix-0225/
python olmocr/train/prepare_olmocrmix.py --subset 01_books --split train_iabooks --destination ~/olmOCR-mix-0225/
python olmocr/train/prepare_olmocrmix.py --subset 00_documents --split eval_s2pdf --destination ~/olmOCR-mix-0225/
python olmocr/train/prepare_olmocrmix.py --subset 00_documents --split train_s2pdf --destination ~/olmOCR-mix-0225/
```

### Setup your config

[olmOCR-7B-0725-FP8](https://huggingface.co/allenai/olmOCR-7B-0725-FP8) was trained with [qwen25_vl_olmocrv2_2epoch.yaml](/olmcr/train/configs/qwen25_vl_olmocrv2_2epoch.yaml)

This is setup to train on a single B200 GPU, and training will take around 48 hours (~$300 if renting). 
Single epoch runs will take half the time and will only lose ~1 point on olmOCR-bench.

But this is training for ~250,000 pages per epoch, so it's quite a big endeavour. We hope to add more options to make further finetuning your own small model more simple and easy.

### Launch training

```bash
python -m olmocr.train.train --config olmcr/train/configs/qwen25_vl_olmocrv2_2epoch.yaml
```

### Prepare Checkpoints and Quantize

After training is done, you will need to call `prepare_checkpoint.py` to take the saved checkpoints
and get them ready for use with VLLM.

```bash
python -m olmocr.train.prepare_olmocr_checkpoint [source dir]/checkpoint-7648 [destination]
```

And finally, we recommend doing an FP8 quantization step, whose performance is solidly in the error bars of the raw
bfloat16 model, but uses less memory and inferences around 12% faster.

```bash
python -m olmocr.train.compress_checkpoint --config olmocr/train/quantization_configs/qwen2_5vl_w8a8_fp8.yaml [destination] [destination-FP8]
```

### Notes for AI2
If you are a collaborator of AI2, you can use the following scripts to run training and inference

```bash
# Run training using Beaker
scripts/train/newtrainer-beaker.sh --config [config file]

# Prepare checkpoint from an interactive session with WEKA
python -m olmocr.train.prepare_olmocr_checkpoint [source] [destination]

# Compress the prepared model checkpoint to FP8
scripts/train/compress_model.sh <recipe_path> <input_model_path> <output_model_path>[--calibration-pdfs PATTERN]

# Run olmOCR bench
scripts/run_benchmark.sh --model [destination]
```
