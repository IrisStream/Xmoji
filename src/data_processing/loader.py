import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import torch
import numpy as np
import pandas as pd

# class EmojiDataset(Dataset):
#     def __init__(self, json_file):
#         self.data = self.load_data(json_file)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         item = self.data[index]
#         label = item['label']
#         sentence = item['sentence']
#         return {'label': label, 'sentence': sentence}

#     def load_data(self, json_file):
#         with open(json_file, 'r') as file:
#             data = json.load(file)
#         return data

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

class EmojiDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.data.TEXT[index])
        sentence = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

def get_loader(file_path: str, batch_size: int=32, shuffle: bool=True, split_supp_query:bool = True) -> DataLoader:
    # load data from file into torch.Tensor
    df = pd.read_csv(file_path, sep='\t', names=["TEXT", "LABEL"])

    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': 0
    }

    if split_supp_query:
        query_df=df.sample(frac=0.8,random_state=200)
        support_df=df.drop(query_df.index).reset_index(drop=True)
        query_df = query_df.reset_index(drop=True)

        support_set, query_set = EmojiDataset(support_df, tokenizer, MAX_LEN), EmojiDataset(query_df, tokenizer, MAX_LEN)
        return DataLoader(support_set, **dataloader_params), DataLoader(query_set, **dataloader_params)
    else:
        return DataLoader(EmojiDataset(df, tokenizer, MAX_LEN), **dataloader_params)
