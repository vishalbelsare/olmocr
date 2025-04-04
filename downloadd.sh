#!/bin/bash
while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        echo "Downloading $line..."
        aws s3 cp "$line" .
    fi
done < s2pdf_paths.txt