import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import json
import shutil
from settings import strategies, FL_config, data_info

# ok
# def get_X_y():
#     df = data_info.get_table()

#    df = pd.read_csv(file_path, sep='\t',  encoding="utf-8", names=["TEXT", "LABEL"])
#     encode_dict = {}

#     def encode_emoji(x):
#         if x not in encode_dict.keys():
#             encode_dict[x]=len(encode_dict)
#         return encode_dict[x] - 1

#     df['ENCODE_LABEL'] = df['LABEL'].apply(lambda x: encode_emoji(x))

#     X = df["TEXT"].to_numpy()
#     y = df['ENCODE_LABEL'].to_numpy()

#    pd.DataFrame(df[["LABEL", "ENCODE_LABEL"]]).to_json(os.path.join(data_path,"labels.json"), orient='records', lines=True)

#    return X, y


def get_data_by_label(data:pd.DataFrame, num_labels:int) -> list[np.ndarray]:
    data_by_label = []
    for label in range(num_labels):
        data_by_label.append(data[data.LABEL == label])
        # shuffle to have client with different data after generating
        data_by_label[label](frac=1).reset_index(drop=True)
    return data_by_label

# def draw_dis(axis, client:dict[int, list], title:str=None):
#     agg_data = client.groupby('LABEL')['TEXT'].count().reset_index()
#     x = agg_data.LABEL.to_numpy()
#     y = agg_data.TEXT.to_numpy()
#     axis.bar(x, y)
#     if title:
#         axis.set_title(title)

# def vis_dis_client(dir:str, dataset:str, k:int):
#     train_dir = os.path.join(dir, dataset, f'client_train_{k}')
#     test_dir  = os.path.join(dir, dataset, f'client_test_{k}')

#     fig, ax = plt.subplots(nrows=8, ncols=10, figsize=(15, 15), sharey=True)
#     fig.subplots_adjust(hspace=0.5)
#     list_file = os.listdir(train_dir)
#     list_file.sort()
#     for idx, file_ in enumerate(list_file):
#         fi = open(os.path.join(train_dir, file_), 'r')
#         data = pd.read_csv(fi, sep='\t', names=data_info.COLUMN_NAMES)
#         draw_dis(ax.flat[idx], data, file_)
#     fig.suptitle(f'train {dataset}: k={k}')
#     fig.show()

#     fig, ax = plt.subplots(nrows=4, ncols=10, figsize=(15, 7), sharey=True)
#     fig.subplots_adjust(hspace=0.5)
#     list_file = os.listdir(test_dir)
#     list_file.sort()
#     for idx, file_ in enumerate(list_file):
#         fi = open(os.path.join(test_dir, file_), 'r')
#         data = pd.read_csv(fi, sep='\t', names=data_info.COLUMN_NAMES)
#         draw_dis(ax.flat[idx], data, file_)
#     fig.suptitle(f'test {dataset}: k={k}')
#     fig.show()

def write_data(file_path:str, client:dict[int, pd.DataFrame]):
    df = pd.concat(client.values(), ignore_index=True)
    df.to_csv(file_path, index=False, sep='\t')

def write_client(dir:str, id:int, client:dict[int, pd.DataFrame], split_supp_qry:bool=False):
    if split_supp_qry:
        supp_set, qry_set = {}, {}
        for label in client.keys():
            num_samples = len(client[label])
            qry_set[label] = client[label].iloc[:int(num_samples*FL_config.QUERY_RATIO), :]
            supp_set[label] = client[label].iloc[int(num_samples*FL_config.QUERY_RATIO):, :]

        test_supp_file = os.path.join(dir, f'{id}_s.csv')
        write_data(test_supp_file, supp_set)
        test_qry_file = os.path.join(dir, f'{id}_q.csv')
        write_data(test_qry_file, qry_set)
    else:
        train_file = os.path.join(dir, f'{id}.csv')
        write_data(train_file, client)

def write_all_clients(clients:dict[int, dict], data_path:str, strategy:str, num_clients:int=100):
    train_data_path = os.path.join(data_path, f'client_train_{strategy}')
    test_data_path = os.path.join(data_path, f'client_test_{strategy}')
    if os.path.isdir(train_data_path):
        shutil.rmtree(train_data_path)
    if os.path.isdir(test_data_path):
        shutil.rmtree(test_data_path)
    os.mkdir(train_data_path)
    os.mkdir(test_data_path)

    print(f'Write data to {train_data_path} and {test_data_path}\n')
    for client in clients.keys():
        if client < num_clients*FL_config.TRAIN_RATIO:
            # Write TRAIN data
            write_client(dir=train_data_path, id=client, client=clients[client])
        else:
            # Write TEST data
            write_client(dir=test_data_path, id=client, client=clients[client], split_supp_qry=True)

# type = {'uniform', 'dirichlet'}
def distribution_based_split_client(data_by_label:list[np.array], num_clients:int=100, num_labels:int=10, type:str='dirichlet'):
    # compute number of samples for each label
    num_samples_per_label = []
    for data in data_by_label:
        num_samples_per_label.append(len(data))

    num_samples_in_client = []
    # compute the percentages data of each label in each client
    if type=='uniform':
        # split using Uniform distribution
        alpha = np.full(num_clients, 1/num_clients)
        for label in range(num_labels):
            num_samples_in_client.append((np.round(alpha * num_samples_per_label[label]).astype(int)).tolist())
    elif type=='dirichlet':
        # split using Distribution-based label imbalance mode
        alpha = np.full(num_clients, 0.5) # 0.5 from an A* paper
        for label in range(num_labels):
            p = np.random.dirichlet(alpha)
            num_samples_in_client.append((np.round(p * num_samples_per_label[label]).astype(int)).tolist())

    # split data to client
    clients = {}
    counts = {i:0 for i in range(num_labels)}
    for i in range(num_clients):
        clients[i] = {}
        for label in range(num_labels):
            num_samples = num_samples_in_client[label][i]
            if num_samples != 0:
                lower_bound = counts[label]
                upper_bound = lower_bound + num_samples
                clients[i][label] = data_by_label[label].iloc[lower_bound:upper_bound, :]
                counts[label] = upper_bound

    return clients

# Distribution-based label imbalance: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
def dirichlet_based_gen(data:pd.DataFrame, num_clients:int=100, num_labels:int=10):
    # distribution_data_path=os.path.join(data_path, strategies.DISTRIBUTION_BASED_SKEW)
    # if os.path.isdir(distribution_data_path):
    #     shutil.rmtree(distribution_data_path)
    # os.mkdir(distribution_data_path)

    data_by_label = get_data_by_label(data, num_labels)

    print(f'Generate data: Dirichlet distribution, num_clients={num_clients}, num_labels={num_labels}')
    return distribution_based_split_client(data_by_label, num_clients, num_labels)

def iid_gen(data_path='../../data'):
    num_clients = 100
    num_labels = 1573
    k=5 # 5 for iid

    iid_data_path=os.path.join(data_path, 'iid')
    if os.path.isdir(iid_data_path):
        shutil.rmtree(iid_data_path)
    os.mkdir(iid_data_path)

    X, y = get_X_y(data_path)
    data_by_label = get_data_by_label(X, y, num_labels)

    print(f'Generate data: IID, num_clients={num_clients}, num_labels={num_labels}')
    all_clients = distribution_based_split_client(data_by_label, num_clients, num_labels, 'uniform')
    write_all_clients(all_clients, iid_data_path, k)

# split using Quantity-based label imbalance mode
def quantity_based_split_client(data_by_label:list[np.array], k:int, num_clients:int=100, num_labels:int=10):
    partition_labels = {i:0 for i in range(num_labels)}

    # init all clients and partition labels
    all_clients = {}
    for i in range(num_clients):
        all_clients[i] = {}
        labels = np.random.choice(list(range(num_labels)), k, replace=False).tolist()
        for label in labels:
            all_clients[i][label] = None
            partition_labels[label] += 1

    # split sample of label into partitions
    labels_after_partition = {i:None for i in range(num_labels)}
    counts = {i:0 for i in range(num_labels)}
    for i in range(num_labels):
        if partition_labels[i] != 0:
            labels_after_partition[i] = np.array_split(data_by_label[i], partition_labels[i])

    # split data into clients
    for i in range(num_clients):
        for label in all_clients[i].keys():
            all_clients[i][label] = labels_after_partition[label][counts[label]].tolist()
            counts[label] += 1

    return all_clients

# Quantity-based label imbalance: each party owns data samples of a fixed number of labels.
def quantity_base_gen(data_path='../../data'):
    ks = [15,30,45] # 1,2,3 for quantity-based imbalance
    num_clients = 100
    num_labels = 1573

    # quantity_data_path=os.path.join(data_path, 'quantity')
    # if os.path.isdir(quantity_data_path):
    #     shutil.rmtree(quantity_data_path)
    # os.mkdir(quantity_data_path)

    X, y = get_X_y(data_path)
    data_by_label = get_data_by_label(X, y, num_labels)

    for k in ks:
        print(f'Generate data: k={k}, num_clients={num_clients}, num_labels={num_labels}')
        all_clients = quantity_based_split_client(data_by_label, k, num_clients, num_labels)

        # Write data to quantity directory
        write_all_clients(
            all_clients=all_clients, 
            data_path=data_path,
            strategy=f'quantity',
            num_clients=num_clients
        )