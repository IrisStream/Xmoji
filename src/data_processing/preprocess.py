import os
import json
import emoji as em
from typing import *
from nltk.tokenize import TweetTokenizer
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Function to remove skin tone modifiers from a given string
def strip_emoji_modifiers(emoji: str) -> str:
    '''
    Remove skin tone modifiers from the input string.

    :param emoji: The input string containing emojis with or without skin tone modifiers
    :return: A copy of the input string with skin tone modifiers removed
    '''

    return emoji[0]


async def process_file(file_name):
    input_file_path = os.path.join('../../data/normalized_input', file_name)
    
    # Asynchronously read content from the input file
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, read_file_content, input_file_path)
    
    texts = em.replace_emoji(content, replace=lambda chars, data_dict: f'\t{strip_emoji_modifiers(chars)}\n').strip()
    
    # Generate the output file name based on the input file name
    output_file_path = os.path.join('../../data/output', file_name)
    
    # Asynchronously write content to the specific output file
    await loop.run_in_executor(None, write_to_output, output_file_path, texts)

def read_file_content(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_to_output(output_file_path, texts):
    with open(output_file_path, 'a') as output_file:
        output_file.write(texts + "\n")

async def main():
    folder_path = '../../data/normalized_input'
    files = os.listdir(folder_path)

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:
        # Process each file asynchronously
        await asyncio.gather(*[process_file(file) for file in files])

if __name__ == "__main__":
    # with open('../../data/input.jsonl', 'r') as input_file:
    #     for i, line in enumerate(input_file):
    #         # Load the JSON object from the line
    #         json_object = json.loads(line)

    #         # Extract the "text" property from the JSON object
    #         tweets = json_object.get('text', [])
            
    #         with open(f'../../data/raw_input/{i}.txt', 'w') as output_file:
    #             output_file.write('\n'.join(tweets))

    # try:
    #     subprocess.run(['bash', 'normalize_tweets.sh', '../../data/raw_input', '../../data/normalized_input'], check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error: {e}")

    asyncio.run(main())