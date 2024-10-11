#!/bin/bash

# Directory containing the files
DIRECTORY="./datasets"  # Change this to your target directory
PYTHON_PROGRAM="./src/life_watch.py"      # Change this to your Python script name

BASH_START_TIME=$(date +%s)
# Iterate over all files in the specified directory
for FILE in "$DIRECTORY"/*; do
    # Check if it's a file
    if [[ -f "$FILE" ]]; then
        echo "Processing file: $FILE"
        # Start time
        START_TIME=$(date +%s)

        # Call the Python program with the file as an argument
        python "$PYTHON_PROGRAM" "-i" "$FILE"

        # End time
        END_TIME=$(date +%s)
        
        # Calculate duration
        DURATION=$((END_TIME - START_TIME))
        
        echo "Time taken to process $FILE: $DURATION seconds"
    else
        echo "$FILE is not a valid file, skipping."
    fi
done

BASH_END_TIME=$(date +%s)

DURATION=$((BASH_END_TIME - BASH_START_TIME))
        
echo "Time taken to process all tasks: $DURATION seconds"
