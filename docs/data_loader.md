# Data loader

## *EmojiDataset* Dataset Class
- This class is defined to accept the Dataframe as input and generate tokenized output that is used by the DistilBERT model for training. 
- We are using the DistilBERT tokenizer to tokenize the data in the `TEXT` column of the dataframe. 
- The tokenizer uses the `encode_plus` method to perform tokenization and generate the necessary outputs, namely: `ids`, `attention_mask`
- `targets` is the encoded category on the news headline. 

## Dataloader
- Dataloader is used to for creating training and validation dataloader that load data to the neural network in a defined manner. This is needed because all the data from the dataset cannot be loaded to the memory at once, hence the amount of dataloaded to the memory and then passed to the neural network needs to be controlled.
- This control is achieved using the parameters such as `batch_size` and `max_len`.
- Training and Validation dataloaders are used in the training and validation part of the flow respectively

## *get_loader* function
- Return a Dataloader object according to:
	- `file_name`: the path to the csv file 
	- `batch_size`: the batch size value for `Dataloader`
	- `shuffle`: with shuffle the data in dataset if `True`
	- `split_supp_query`: if `False` return 1 dataloader
		- Split the dataset into 2 datasets (Support & Query)
		- Support:Querry = 2:8
		- Return 2 Dataloader for each 
		- Used to create dataloader for **Train** client

