#!/bin/bash -x

# Compile the C++ code for filter the datas
g++ -o preprocess filter_valid_data.cpp

# Need to run split_dir.sh to split the data (OS can't handle too many file director)
for i in {1..20}; do
	echo "OUTPUT$i"
	./preprocess ../../data/output$i ../../data/normalized_output
done