import pandas as pd 
import numpy as np
import os 
import csv
import matplotlib.pyplot as plt

class strategies:
    QUANTITY_BASED_SKEW = "quantity"
    DISTRIBUTION_BASED_SKEW = "dirichlet"
    IID = "iid"

class FL_config:
    NUM_CLIENTS = 100
    QUERY_RATIO = 0.8
    TRAIN_RATIO = 0.8
    DIRICHLET_RATIO = 0.5

class Data:
    __instance = None 

    def __init__(self):
        self.__data_path = "../../data"
        self.__file_name = "data.csv"
        self.__frequency_file_name = 'label.csv'
        self.__columns = ["TEXT", "LABEL"]
        # self.__min_frequency = 5000
        # self.__scale = 1
        self.__min_frequency = 8000
        self.__scale = 0.1
        self.__table, self.__frequency_table, self.__num_labels = self.__load_table()
        self.__words_count_per_sample = self.__compute_words_count_per_sample()
        self.__data_statistic = self.__compute_data_statistic()

    @staticmethod
    def get_data_statistic() -> dict:
        return Data.__get_instance().__data_statistic


    @staticmethod
    def get_columns():
        return Data.__get_instance().__columns

    @staticmethod
    def get_data_path():
        return Data.__get_instance().__data_path

    @staticmethod
    def get_num_labels():
        return Data.__get_instance().__num_labels
    
    @staticmethod
    def get_table():
        return Data.__get_instance().__table

    @staticmethod
    def get_num_frequency():
        return Data.__get_instance().__min_frequency

    @staticmethod
    def get_frequency_table():
        return Data.__get_instance().__frequency_table
    
    @staticmethod 
    def write_frequency_data():
        instance = Data.__get_instance()
        file_path = os.path.join(instance.__data_path, instance.__frequency_file_name)
        instance.__frequency_table.to_csv(file_path, index=False)

    @staticmethod 
    def draw_frequncy_bar_chart():
        plt.figure(figsize=(15,6))
        Data.get_frequency_table().Frequency.plot(kind='bar')
        plt.title('Emoji Frequency Bar Chart')
        plt.xlabel('Emoji')
        plt.ylabel('Frequency')
        plt.xticks([])
        plt.show()

    # TODO:
    # 1. Load input data
    # 2. Get emoji frequency list >= MIN_FREQUENCY
    # 3. Filter out the data with list #2
    # 4. Encode the emoji in data table
    # 5. Create data_by_label table
    def __load_table(self) -> tuple[list[pd.DataFrame], pd.DataFrame, int]:
        # 1. Load input data
        file_path = os.path.join(self.__data_path, self.__file_name)
        df = pd.read_csv(file_path, sep='\t',encoding='utf-8', quoting=csv.QUOTE_NONE, names=self.__columns, header=0)
        
        # 2. Get emoji frequency list >= MIN_FREQUENCY
        emoji_frequency_list = self.__compute_labels_frequency(df, self.__min_frequency, self.__scale)
        num_labels = len(emoji_frequency_list)

        # 3. Filter out the data with list #2
        filtered_df = df[df.LABEL.isin(emoji_frequency_list.emoji)]

        # 4. Encode the emoji in data table
        encoded_df = pd.merge(filtered_df, emoji_frequency_list, left_on='LABEL', right_on='emoji')[['TEXT', 'encoded_label']]
        encoded_df.rename(columns={'encoded_label':'LABEL'}, inplace=True)

        # 5. Create data_by_label table
        data_by_label = Data.__create_data_by_label(encoded_df, num_labels, emoji_frequency_list)

        return data_by_label, emoji_frequency_list, num_labels
    
    @staticmethod
    def __compute_labels_frequency(df:pd.DataFrame, min_frequency:int, scale:int):
        emoji_counts = df.LABEL.value_counts()
        emoji_counts = emoji_counts[emoji_counts >= min_frequency]
        emoji_counts = emoji_counts.rename_axis('emoji').reset_index()
        emoji_counts = emoji_counts.rename_axis('encoded_label').reset_index()
        emoji_counts.rename(columns={'LABEL':'Frequency'}, inplace=True)
        emoji_counts.Frequency = emoji_counts.Frequency.apply(lambda x : int(x * scale))
        return emoji_counts

    @staticmethod
    def __create_data_by_label(data:pd.DataFrame, num_labels:int, emoji_frequency_list:pd.DataFrame) -> list[pd.DataFrame]:
        data_by_label = []
        for label in range(num_labels):
            data_by_label.append(data[data.LABEL == label].sample(frac=1).reset_index(drop=True).head(emoji_frequency_list.Frequency[label]))
            # rows_to_select = int(scale * len(data_by_label[-1]))
            # data_by_label[-1] = data_by_label[-1].iloc[:rows_to_select, :]
            # shuffle to have client with different data after generating
            # data_by_label[label](frac=1).reset_index(drop=True)
        return data_by_label
    
    @staticmethod 
    def __get_instance():
        if Data.__instance == None:
            Data.__instance = Data()
        return Data.__instance

    def __compute_data_statistic(self):
        data_stt = {
            "Total samples": self.__get_total_samples(),
            "Total classes": self.__num_labels,
            "Max words per sample": self.__get_max_words_per_sample(),
            "Mean words per sample": self.__get_mean_words_per_sample(),
            "Std words per sample": self.__get_std_words_per_sample()
        }
        df = pd.DataFrame(columns = ["Key", "Value"], data = list(data_stt.items()))
        file_path = os.path.join(self.__data_path, 'data_statistic.csv')
        df.to_csv(file_path, index=False)
        return data_stt

    def __get_total_samples(self):
        return self.__frequency_table.Frequency.sum()

    def __compute_words_count_per_sample(self):
        df = pd.concat(self.__table, ignore_index=True)
        words_count = df.TEXT.apply(lambda x : len(x.split()))
        return words_count

    def __get_words_per_sample(self):
        return self.__words_count_per_sample

    def __get_mean_words_per_sample(self):
        words_count = self.__get_words_per_sample()
        return words_count.mean()

    def __get_std_words_per_sample(self):
        words_count = self.__get_words_per_sample()
        return words_count.std()

    def __get_max_words_per_sample(self):
        words_count = self.__get_words_per_sample()
        return words_count.max()
