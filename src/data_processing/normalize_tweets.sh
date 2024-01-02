#!/bin/bash

input_dir=$1
output_dir=$2

for file_path in "$input_dir"/*; do
	file_name=$(basename "$file_path" .txt)
	awk '{
		line = tolower($0);
		gsub(/’/, "\x27", line);
		gsub(/…/, "...", line);
		gsub(/cannot /, "can not ", line);
		gsub(/n'\''t /, " n'\''t ", line);
		gsub(/n '\''t /, " n'\''t ", line);
		gsub(/ca n'\''t/, "can'\''t", line);
		gsub(/ai n'\''t/, "ain'\''t", line);
		gsub(/'\''m /, " '\''m ", line);
		gsub(/'\''re /, " '\''re ", line);
		gsub(/'\''s /, " '\''s ", line);
		gsub(/'\''ll /, " '\''ll ", line);
		gsub(/'\''d /, " '\''d ", line);
		gsub(/'\''ve /, " '\''ve ", line);
		gsub(/ p\. m\./, " p.m.", line);
		gsub(/ p\. m /, " p.m ", line);
		gsub(/ a\. m\./, " a.m.", line);
		gsub(/ a\. m /, " a.m ", line);
		gsub(/@[a-zA-Z0-9_]+/, "@USER", line); # Replace user tags starting with '@'
		gsub(/(http[^[:space:]]+|www[^\s]+)/, "HTTPURL", line); # Adjusted URL pattern
		print line;
	}' $file_path > $output_dir/$file_name.txt
done