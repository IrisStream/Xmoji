#!/bin/bash

# Create 10 destination directories
for i in {1..20}; do mkdir -p "../../data/output$i"; done

# Use find, xargs, and mv to distribute files without randomness
counter=1
find ../../data/output -type f -print0 | while IFS= read -r -d '' file; do
    mv "$file" "../../data/output$counter"
    counter=$(( (counter % 20) + 1 ))
done
