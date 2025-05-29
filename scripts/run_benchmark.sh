#!/bin/bash

set -e

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

# Create full image tag
IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

# Get Beaker username
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
echo "Beaker user: $BEAKER_USER"

# Push image to beaker
echo "Pushing image to Beaker..."
beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_benchmark_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskContext, ResultSpec, TaskResources, ImageSource, Priority, Constraints

# Get image tag and beaker user from command line
image_tag = sys.argv[1]
beaker_user = sys.argv[2]

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/olmocr")

# Create experiment spec
experiment_spec = ExperimentSpec(
    description="OlmOCR Benchmark Run",
    budget="ai2/oe-data",
    tasks=[
        TaskSpec(
            name="olmocr-benchmark",
            image=ImageSource(beaker=f"{beaker_user}/{image_tag}"),
            command=[
                "bash", "-c",
                " && ".join([
                    "git clone https://huggingface.co/datasets/allenai/olmOCR-bench",
                    "cd olmOCR-bench && git lfs pull && cd ..",
                    "python -m olmocr.pipeline ./localworkspace --markdown --pdfs ./olmOCR-bench/bench_data/pdfs/**/*.pdf",
                    "python -m olmocr.bench.convert olmocr_pipeline --dir ./olmOCR-bench/bench_data",
                    "python olmocr/bench/scripts/workspace_to_bench.py localworkspace/ olmOCR-bench/bench_data/olmocr_full_pipeline --bench-path ./olmOCR-bench/",
                    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data"
                ])
            ],
            context=TaskContext(
                priority=Priority.normal,
                preemptible=True,
            ),
            resources=TaskResources(gpu_count=1),
            constraints=Constraints(cluster=["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2"]),
            result=ResultSpec(path="/noop-results"),
        )
    ],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/olmocr")
print(f"Created experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."
$PYTHON /tmp/run_benchmark_experiment.py $IMAGE_TAG $BEAKER_USER

# Clean up temporary file
rm /tmp/run_benchmark_experiment.py

echo "Benchmark experiment submitted successfully!"