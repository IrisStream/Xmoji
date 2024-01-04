#!/bin/bash

# Define the function to process a single file
process_file() {
	input_file_path=$1
    file_name=$(basename "$input_file_path" .txt)
    # input_file_path="../../data/normalized_input/$file_name"
    
    # Read content from the input file
    content=$(cat "$input_file_path")
    
    # Process content and replace emojis
    # texts=$(echo "$content" | python3 -c "import sys, emoji; print(emoji.replace_emoji(sys.stdin.read(), replace=lambda chars, data_dict: f'\t{chars[0]}\n').strip())")
    texts=$(python3 -c "import sys, emoji; print(emoji.replace_emoji(sys.stdin.read(), replace=lambda chars, data_dict: f'\t{chars[0]}\n').strip())" <<< "$content")


    # echo $input_file_path
    # echo $texts 
    # Generate the output file name based on the input file name
    output_file_path="../../data/output/$file_name"
    
    # Write content to the specific output file
    echo -e "$texts" > "$output_file_path"
}

# Define the function to run the script for each file
# run_script() {
folder_path="../../data/normalized_input"
# files=("$folder_path"/*)
export -f process_file
find "$folder_path" -type f -name '*.txt' | parallel process_file

    # for file in "${files[@]}"; do
    #     file_name=$(basename "$file")
	# 	echo $file_name
    #     process_file "$file_name" &
    # done

    # Wait for all background jobs to finish
    # wait
# }

# Run the script
# run_script
