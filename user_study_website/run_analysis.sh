#!/bin/bash

# This script demonstrates the full workflow for processing Google Form responses
# It first decodes the responses and then runs the analysis

# Check if the input file was provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <responses_file.txt> [output_directory]"
  exit 1
fi

RESPONSES_FILE=$1
OUTPUT_DIR=${2:-"plots"}  # Default to "plots" if not specified

# Step 1: Decode the responses
echo "Decoding responses from $RESPONSES_FILE..."
python decode_responses.py "$RESPONSES_FILE"

# Get the base filename without extension
BASENAME=$(basename -- "$RESPONSES_FILE")
BASENAME="${BASENAME%.*}"
CSV_FILE="${BASENAME}.csv"

# Step 2: Analyze the responses
echo "Analyzing responses and generating visualizations in $OUTPUT_DIR..."
python analyze_responses.py "$CSV_FILE" "$OUTPUT_DIR"

echo "Analysis complete! Results saved to $OUTPUT_DIR directory."
echo "To view the results, open the image files in the $OUTPUT_DIR directory." 