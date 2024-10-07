#!/bin/zsh

# Directory containing the files
DIRECTORY="./datasets"  # Change this to your target directory
PYTHON_PROGRAM="./src/x_watch.py"      # Change this to your Python script name

# Iterate over all files in the specified directory
for FILE in "$DIRECTORY"/*; do
    # Check if it's a file
    if [[ -f "$FILE" ]]; then
        echo "Processing file: $FILE"
        # Call the Python program with the file as an argument
        python "$PYTHON_PROGRAM" "-i" "$FILE"
    else
        echo "$FILE is not a valid file, skipping."
    fi
done
