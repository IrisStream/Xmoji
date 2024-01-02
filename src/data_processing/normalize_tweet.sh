#!/bin/bash

input=$1
output=$2
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
    gsub(/@[a-zA-Z0-9_]+/, "USER", line); # Replace user tags starting with '@'
    gsub(/(http[^[:space:]]+|www[^\s]+)/, "URL", line); # Adjusted URL pattern
    print line;
}' $input > $output