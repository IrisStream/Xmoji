import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import json
import shutil
import csv
from settings import strategies, FL_config, Data

def draw_dis(axis, file_path:str, title:str=None):
    data = pd.read_csv(file_path, sep='\t',quoting=csv.QUOTE_NONE, names=Data.get_columns(), header=0)
    dis = data.LABEL.value_counts().sort_index()
    x = dis.index
    y = dis.values
    axis.bar(x, y, width=2)
    # axis.get_xaxis().set_visible(False)
    axis.set_title(title)
    return dis.max()

def draw_all_dis(data_path:str, fig_size, title:str):
    list_file = os.listdir(data_path)
    list_file.sort()

    fig, ax = plt.subplots(nrows=len(list_file)//10, ncols=10, figsize=fig_size, sharey=True)
    fig.subplots_adjust(hspace=0.5)

    ylimit = np.zeros(len(list_file))
    for idx, file_ in enumerate(list_file):
        file_path = os.path.join(data_path, file_)
        ylimit[idx] = draw_dis(ax.flat[idx], file_path, file_)
        # print(f'{file_path} - {ylimit[idx]}')

    fig.suptitle(title)
    plt.setp(ax, xlim=(-4, Data.get_num_labels() + 4), ylim=(0, ylimit.mean()))
    fig.tight_layout() 
    fig.show()

# def vis_dis_client(dir:str, dataset:str, k:int):
def vis_dis_client(data_path:str, strategy:str):
    train_data_path, test_data_paths = get_output_data_path(data_path, strategy)

    draw_all_dis(train_data_path, (15, 15), f'train {strategy}')
    draw_all_dis(test_data_paths['query'], (15, 4), f'test query {strategy}')
    draw_all_dis(test_data_paths['support'], (15, 4), f'test support {strategy}')

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

        test_supp_file = os.path.join(dir, 'support', f'{id}_s.csv')
        write_data(test_supp_file, supp_set)
        test_qry_file = os.path.join(dir, 'query', f'{id}_q.csv')
        write_data(test_qry_file, qry_set)
    else:
        train_file = os.path.join(dir, f'{id}.csv')
        write_data(train_file, client)

def get_output_data_path(data_path:str, strategy:str) -> tuple[str, dict]:
    train_data_path = os.path.join(data_path, f'{strategy}_client_train')
    test_data_path = os.path.join(data_path, f'{strategy}_client_test')
    test_data_paths = {
        'all': test_data_path,
        'query': os.path.join(test_data_path, 'query'),
        'support': os.path.join(test_data_path, 'support')
    }
    return train_data_path, test_data_paths

def reset_data(data_path:str, strategy: str) -> tuple[str, dict]:
    train_data_path, test_data_paths = get_output_data_path(data_path, strategy)
    if os.path.isdir(train_data_path): shutil.rmtree(train_data_path)
    if os.path.isdir(test_data_paths['all']): shutil.rmtree(test_data_paths['all'])
    os.mkdir(train_data_path)
    os.makedirs(test_data_paths['query'])
    os.makedirs(test_data_paths['support'])
    return train_data_path, test_data_paths

def write_all_clients(clients:dict[int, dict], data_path:str, strategy:str):
    train_data_path, test_data_paths = reset_data(data_path, strategy)

    print(f"Write data to {train_data_path} and {test_data_paths['all']}\n")
    num_clients = len(clients)
    for client in clients.keys():
        if client < num_clients*FL_config.TRAIN_RATIO:
            write_client(dir=train_data_path, id=client, client=clients[client])
        else:
            write_client(dir=test_data_paths['all'], id=client, client=clients[client], split_supp_qry=True)

# type = {'uniform', 'dirichlet'}
def distribution_based_split_client(data_by_label:list[pd.DataFrame], num_clients:int, num_labels:int, type:str):
    # compute number of samples for each label
    num_samples_per_label = []
    for data in data_by_label:
        num_samples_per_label.append(len(data))

    num_samples_in_client = []
    # compute the percentages data of each label in each client
    if type=='uniform':
        alpha = np.full(num_clients, 1/num_clients)
        # split using Uniform distribution
        for label in range(num_labels):
            num_samples_in_client.append((np.round(alpha * num_samples_per_label[label]).astype(int)).tolist())
    elif type=='dirichlet':
        # split using Distribution-based label imbalance mode
        alpha = np.full(num_clients, FL_config.DIRICHLET_RATIO)
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
def dirichlet_based_gen():
    data_by_label = Data.get_table()
    num_labels = Data.get_num_labels()
    num_clients = FL_config.NUM_CLIENTS

    print(f'Generate data: Dirichlet distribution, num_clients={num_clients}, num_labels={num_labels}')
    return distribution_based_split_client(data_by_label, num_clients, num_labels, strategies.DISTRIBUTION_BASED_SKEW)

def iid_gen():
    data_by_label = Data.get_table()
    num_labels = Data.get_num_labels()
    num_clients = FL_config.NUM_CLIENTS


    print(f'Generate data: IID, num_clients={num_clients}, num_labels={num_labels}')
    return distribution_based_split_client(data_by_label, num_clients, num_labels, 'uniform')

# split using Quantity-based label imbalance mode
def quantity_based_split_client(data_by_label:list[pd.DataFrame], labels_per_client:int, num_clients:int, num_labels:int):
    partition_labels = {i:0 for i in range(num_labels)}

    # init all clients and partition labels
    clients = {}
    for i in range(num_clients):
        clients[i] = {}
        labels = np.random.choice(list(range(num_labels)), labels_per_client, replace=False).tolist()
        for label in labels:
            clients[i][label] = None
            partition_labels[label] += 1

    # split sample of label into partitions
    labels_after_partition = {i:None for i in range(num_labels)}
    counts = {i:0 for i in range(num_labels)}
    for i in range(num_labels):
        if partition_labels[i] != 0:
            labels_after_partition[i] = np.array_split(data_by_label[i], partition_labels[i])

    # split data into clients
    for i in range(num_clients):
        for label in clients[i].keys():
            clients[i][label] = labels_after_partition[label][counts[label]]
            counts[label] += 1

    return clients

# Quantity-based label imbalance: each party owns data samples of a fixed number of labels.
def quantity_based_gen(labels_per_client:int):
    data_by_label = Data.get_table()
    num_labels = Data.get_num_labels()
    num_clients = FL_config.NUM_CLIENTS

    print(f'Generate data: Dirichlet distribution, num_clients={num_clients}, num_labels={num_labels}')
    return quantity_based_split_client(data_by_label, labels_per_client, num_clients, num_labels)

def data_statistic(clients, strategy):
    samples_per_user = np.array([len(pd.concat(client.values(), ignore_index=True)) for client in clients.values()])
    data_stt = {
        "Total clients" : FL_config.NUM_CLIENTS,
        "Mean samples per user": np.mean(samples_per_user),
        "Std samples per user": np.std(samples_per_user)
    }
    df = pd.DataFrame(columns = ["Key", "Value"], data = list(data_stt.items()))
    file_path = os.path.join(Data.get_data_path(), f'statistic_{strategy}.csv')
    df.to_csv(file_path, index=False)

def data_generator(output_data_path:str, strategy:str, *arg):
    if strategy == strategies.DISTRIBUTION_BASED_SKEW:
        clients = dirichlet_based_gen()
    elif strategy == strategies.QUANTITY_BASED_SKEW:
        labels_per_client = arg[0]
        strategy = f"{strategy}_{labels_per_client}"
        clients = quantity_based_gen(labels_per_client)
    elif strategy == strategies.IID:
        clients = iid_gen()
    
    write_all_clients(clients, output_data_path, strategy)
    data_statistic(clients, strategy)