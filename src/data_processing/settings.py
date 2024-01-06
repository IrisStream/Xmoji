import pandas as pd 
import os 


class strategies:
	QUANTITY_BASED_SKEW = "quantity"
	DISTRIBUTION_BASED_SKEW = "dirichlet"
	IID = "iid"

class FL_config:
	NUM_CLIENTS = 100
	QUERY_RATIO = 0.8
	TRAIN_RATIO = 0.8

class data_info:
	# public properties
	DATA_PATH = "../../data"
	FILE_NAME = "data.csv"
	COLUMN_NAMES = ["TEXT", "LABEL"]

	# private properties
	_TABLE = None
	_NUM_LABELS = 0

	@classmethod
	def get_num_labels():
		if data_info._NUM_LABELS == 0:
			data_info._NUM_LABELS = data_info._compute_num_labels() 
		return data_info._NUM_LABELS
	
	@classmethod
	def _compute_num_labels():
		file_path = os.path.join(data_info.DATA_PATH, data_info.FILE_NAME)
		df = pd.read_csv(file_path, sep='\t', names=data_info.COLUMN_NAMES)
		return df.LABEL.nunique()

	@classmethod
	def get_table():
		if data_info._TABLE == None:
			data_info._TABLE = data_info._load_table()
		return data_info._TABLE


	@classmethod
	def _load_table():
		file_path = os.path.join(data_info.DATH_PATH, data_info.FILE_NAME)
		df = pd.read_csv(file_path, sep='\t', names=data_info.COLUMN_NAMES)

		encode_dict = {}

		def encode_emoji(x):
			if x not in encode_dict.keys():
				encode_dict[x]=len(encode_dict)
			return encode_dict[x] - 1

		df['ENCODED_LABEL'] = df['LABEL'].apply(lambda x: encode_emoji(x))

		new_df = pd.DataFrame(df[['TEXT', 'ENCODED_LABEL']])

		new_df = new_df.rename(columns={'ENCODED_LABEL': 'LABEL'})

		return new_df
		