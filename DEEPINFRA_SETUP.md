# Using olmOCR with DeepInfra

This guide explains how to use olmOCR with DeepInfra's hosted API service for cloud-based inference.

## Prerequisites

1. **DeepInfra Account**: Sign up at https://deepinfra.com/
2. **API Key**: Get your API key from the DeepInfra dashboard
3. **olmOCR**: Ensure you have the modified version with authentication support

## Setup

### 1. Get your DeepInfra API Key

1. Log in to https://deepinfra.com/
2. Navigate to your dashboard
3. Generate or copy your API key
4. Store it securely (recommended: as an environment variable)

```bash
export DEEPINFRA_API_KEY="your-api-key-here"
```

### 2. Usage

Run olmOCR with the DeepInfra server endpoint:

```bash
python -m olmocr.pipeline ./localworkspace \
  --server https://api.deepinfra.com/v1/openai \
  --api_key $DEEPINFRA_API_KEY \
  --model allenai/olmOCR-7B-0725-FP8 \
  --markdown \
  --pdfs path/to/your/*.pdf
```

### Command Line Arguments

- `--server`: DeepInfra's OpenAI-compatible endpoint: `https://api.deepinfra.com/v1/openai`
- `--api_key`: Your DeepInfra API key (or use environment variable)
- `--model`: The model identifier on DeepInfra: `allenai/olmOCR-7B-0725-FP8`
- Other arguments work the same as with local inference

### Example with S3 Storage

For large-scale processing with S3:

```bash
python -m olmocr.pipeline s3://your-bucket/workspace \
  --server https://api.deepinfra.com/v1/openai \
  --api_key $DEEPINFRA_API_KEY \
  --model allenai/olmOCR-7B-0725-FP8 \
  --pdfs s3://your-bucket/pdfs/*.pdf \
  --workers 10 \
  --markdown
```

## Pricing

As of 2024, DeepInfra charges for the olmOCR model:
- Input tokens: ~$0.27 per million tokens
- Output tokens: ~$0.81 per million tokens

Check current pricing at: https://deepinfra.com/pricing