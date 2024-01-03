import os
from ftlangdetect import detect

count=0
for file_name in os.listdir('../../data/normalized_output'):
  file_path = os.path.join('../../data/normalized_output', file_name)
  count+=1
  print(f'[INFO] - {count} - {file_name}')
  with open(file_path, 'r') as input_file:
      # Open the output file in write mode
      with open('../../data/data.csv', 'a') as output_file:
        # output_file.write(f'TEXT\tLABEL\n')
        # Read each line from the file
        for line in input_file:
          try:
            if detect(line.split('\t')[0])['lang'] == "en":
              output_file.write(line)
          except:
            print(f'[ERROR] - {file_name}: {line}')