#!/bin/bash

set -e

# Parse command line arguments
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
SKIP_DOCKER_BUILD=false
PREEMPTIBLE=false
MAX_TRAIN_SAMPLES=""
MAX_EVAL_SAMPLES=""
NUM_EPOCHS=1
LEARNING_RATE="1e-5"
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
USE_WANDB=false
WANDB_PROJECT="olmocr-grpo"
WANDB_RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --skip-docker-build)
            SKIP_DOCKER_BUILD=true
            shift
            ;;
        --preemptible)
            PREEMPTIBLE=true
            shift
            ;;
        --max-train-samples)
            MAX_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --max-eval-samples)
            MAX_EVAL_SAMPLES="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum-steps)
            GRAD_ACCUM_STEPS="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model MODEL_NAME             Model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)"
            echo "  --skip-docker-build            Skip Docker build"
            echo "  --preemptible                  Use preemptible instances"
            echo "  --max-train-samples N          Max training samples"
            echo "  --max-eval-samples N           Max evaluation samples"
            echo "  --num-epochs N                 Number of training epochs (default: 1)"
            echo "  --learning-rate LR             Learning rate (default: 1e-6)"
            echo "  --batch-size N                 Batch size per device (default: 1)"
            echo "  --grad-accum-steps N           Gradient accumulation steps (default: 4)"
            echo "  --use-wandb                    Enable W&B logging"
            echo "  --wandb-project NAME           W&B project name"
            echo "  --wandb-run-name NAME          W&B run name"
            exit 1
            ;;
    esac
done

echo "Model: $MODEL_NAME"
echo "Preemptible: $PREEMPTIBLE"
echo "Use W&B: $USE_WANDB"

# Use conda environment Python if available, otherwise use system Python
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    echo "Using conda Python from: $CONDA_PREFIX"
else
    PYTHON="python"
    echo "Warning: No conda environment detected, using system Python"
fi

# Get version from version.py
VERSION=$($PYTHON -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Get current git branch name
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Git branch: $GIT_BRANCH"

# Create full image tag
IMAGE_TAG="olmocr-grpo-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build and push Docker image if not skipping
if [ "$SKIP_DOCKER_BUILD" = false ]; then
    echo "Building Docker image..."
    docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .
    
    # Push image to beaker
    echo "Trying to push image to Beaker..."
    if ! beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG 2>/dev/null; then
        echo "Warning: Beaker image with tag $IMAGE_TAG already exists. Using existing image."
    fi
else
    echo "Skipping Docker build as requested"
fi

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_grpo_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints, EnvVar, DataMount

# Get parameters from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]
git_branch = sys.argv[3]
git_hash = sys.argv[4]
model_name = sys.argv[5]
preemptible = sys.argv[6] == "true"
max_train_samples = sys.argv[7]
max_eval_samples = sys.argv[8]
num_epochs = sys.argv[9]
learning_rate = sys.argv[10]
batch_size = sys.argv[11]
grad_accum_steps = sys.argv[12]
use_wandb = sys.argv[13] == "true"
wandb_project = sys.argv[14]
wandb_run_name = sys.argv[15]

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Build the training command
commands = [
    # Install dependencies
    "pip install .[train]",
    "pip install trl wandb",
    "pip install transformers==4.55.2",  # Updated for GRPO compatibility
    "pip install flash-attn==2.8.0.post2 --no-build-isolation",
    "pip install vllm==v0.10.1.1",
    "pip install s5cmd",
    
    # Sync the bench data from S3
    "echo 'Syncing bench data from S3...'",
    "mkdir -p /data/olmOCR-bench",
    "s5cmd sync 's3://ai2-oe-data/jakep/olmocr/olmOCR-bench-snapshot-082225/*' /data/olmOCR-bench/",
    
    # Build GRPO training command
    "echo 'Starting GRPO training...'",
]

# Build the python command with all parameters
grpo_cmd = [
    "python -m olmocr.train.grpo_train",
    "--train_bench_data_folder /data/olmOCR-bench/bench_data",
    "--eval_bench_data_folder /data/olmOCR-bench/bench_data",  # Using same data for now
    f"--model_name {model_name}",
    "--output_dir /weka/oe-training-default/jakep/olmocr-grpo-checkpoints",
    f"--num_train_epochs {num_epochs}",
    f"--learning_rate {learning_rate}",
    f"--per_device_train_batch_size {batch_size}",
    f"--per_device_eval_batch_size {batch_size}",
    f"--gradient_accumulation_steps {grad_accum_steps}",
]

# Add optional parameters
if max_train_samples:
    grpo_cmd.append(f"--max_train_samples {max_train_samples}")
if max_eval_samples:
    grpo_cmd.append(f"--max_eval_samples {max_eval_samples}")
if use_wandb:
    grpo_cmd.append("--use_wandb")
    grpo_cmd.append(f"--wandb_project {wandb_project}")
    if wandb_run_name:
        grpo_cmd.append(f"--wandb_run_name {wandb_run_name}")

# Add the GRPO command to the commands list
commands.append(" ".join(grpo_cmd))

# Build task spec
task_spec = TaskSpec(
    name="olmocr-grpo-training",
    image=ImageSource(beaker=f"{beaker_user}/{image_tag}"),
    command=[
        "bash", "-c",
        " && ".join(commands)
    ],
    context=TaskContext(
        priority=Priority.normal,
        preemptible=preemptible,
    ),
    resources=TaskResources(
        gpu_count=1,
        shared_memory="10GiB"
    ),
    constraints=Constraints(cluster=["ai2/titan-cirrascale"]),
    result=ResultSpec(path="/noop-results"),
    env_vars=[
        EnvVar(name="LOG_FILTER_TYPE", value="local_rank0_only"),
        EnvVar(name="OMP_NUM_THREADS", value="8"),
        EnvVar(name="BEAKER_USER_ID", value=beaker_user),
        EnvVar(name="AWS_ACCESS_KEY_ID", secret="ALLENNLP_AWS_ACCESS_KEY_ID"),
        EnvVar(name="AWS_SECRET_ACCESS_KEY", secret="ALLENNLP_AWS_SECRET_ACCESS_KEY"),
        EnvVar(name="WANDB_API_KEY", secret="JAKE_WANDB_API_KEY"),
    ],
    datasets=[
        DataMount.new(mount_path="/weka/oe-data-default", weka="oe-data-default"),
        DataMount.new(mount_path="/weka/oe-training-default", weka="oe-training-default"),
    ]
)

# Create experiment spec
experiment_spec = ExperimentSpec(
    description=f"OlmOCR GRPO Training - Model: {model_name}, Branch: {git_branch}, Commit: {git_hash}",
    budget="ai2/oe-base",
    tasks=[task_spec],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created GRPO training experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker GRPO experiment..."
$PYTHON /tmp/run_grpo_experiment.py \
    "$IMAGE_TAG" \
    "$BEAKER_USER" \
    "$GIT_BRANCH" \
    "$GIT_HASH" \
    "$MODEL_NAME" \
    "$PREEMPTIBLE" \
    "$MAX_TRAIN_SAMPLES" \
    "$MAX_EVAL_SAMPLES" \
    "$NUM_EPOCHS" \
    "$LEARNING_RATE" \
    "$BATCH_SIZE" \
    "$GRAD_ACCUM_STEPS" \
    "$USE_WANDB" \
    "$WANDB_PROJECT" \
    "$WANDB_RUN_NAME"

# Clean up temporary file
rm /tmp/run_grpo_experiment.py

echo "GRPO training experiment submitted successfully!"