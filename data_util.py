#coding=utf-8
from collections import defaultdict
import csv
import tqdm
import numpy as np
import random

def read_file_as_dict(input_path):
    '''
    读取csv文件，并将其保存到字典中。csv文件中有两列，保存的是vocab的字典映射关系
    :param input_path: 要读取得csv文件
    :return: 保存的字典
    '''
    d = {}
    with open(input_path) as input_file:
        reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
        for row in reader:
            d[row['col1']] = int(row['col2'])
    return d

def get_maxlen(*paths):
    '''
    得到输入数据中各项指标的最大长度
    :param paths: 输入数据文件，可以有很多个，train，test，dev数据集
    :return: 各项的最大长度信息，保存在字典maxlen中
    '''
    maxlen = defaultdict(int)
    for path in paths:
        with open(path, 'r') as examples_file:
            fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
            reader = csv.DictReader(examples_file, delimiter='\t', fieldnames=fields)
            for row in reader:
                example = {}
                example['question'] = row['question'].split(' ')
                example['qn_entities'] = row['qn_entities'].split('|')
                example['ans_entities'] = row['ans_entities'].split('|')
                example['sources'] = row['sources'].split('|')
                example['relations'] = row['relations'].split('|')
                example['targets'] = row['targets'].split('|')

                maxlen['question'] = max(len(example['question']), maxlen['question'])
                maxlen['qn_entities'] = max(len(example['qn_entities']), maxlen['qn_entities'])
                maxlen['ans_entities'] = max(len(example['ans_entities']), maxlen['ans_entities'])
                maxlen['sources'] = max(len(example['sources']), maxlen['sources'])
                maxlen['relations'] = maxlen['sources']
                maxlen['targets'] = maxlen['sources']
    return maxlen

def data_loader(data_file, vocab_idx, entity_idx):
    '''
    输入数据的载入函数，读取数据保存在列表中，方便训练时读取操作
    :param data_file: 要读取得文件
    :param vocab_idx: vocab字典映射关系
    :param entity_idx: 实体Entity的字典映射关系
    :return: 返回文件数据，以列表形式，列表中每个元素是文件中的一行，并以字典形式保存各项数据
    '''
    with open(data_file, 'r') as f:
        fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
        reader = csv.DictReader(f, delimiter='\t', fieldnames=fields)
        examples = []
        for line in reader:
            example = {}
            example['question'] = [vocab_idx[word]-1 for word in line['question'].split(' ')]
            example['qn_entities'] = [vocab_idx[word]-1 for word in line['qn_entities'].split('|')]
            example['ans_entities'] = [entity_idx[word]-1 for word in line['ans_entities'].split('|')]
            example['sources'] = [vocab_idx[word]-1 for word in line['sources'].split('|')]
            example['relations'] = [vocab_idx[word]-1 for word in line['relations'].split('|')]
            example['targets'] = [vocab_idx[word]-1 for word in line['targets'].split('|')]
            examples.append(example)

    return examples

def pad(arr, L):
    '''
    对数据进行PAD操作，将arr的长度补全至L，使用0进行填充
    :param arr: 要补全的数组
    :param L: 补全后的长度
    :return: 补全后的数组
    '''
    arr_cpy = list(arr)
    assert (len(arr_cpy) <= L)
    while len(arr_cpy) < L:
        arr_cpy.append(0)
    return arr_cpy

def prepare_batch(batch_examples, maxlen, batch_size, entity_size):
    '''
    对一个batch数据进行填充和扩展
    :param batch_examples: mini-batch的数据
    :param maxlen: 各项指标的最大长度，要进行填充
    :param batch_size: minibatch大小
    :return: 处理完之后的数据，是一个字典，每一项都是一个数组，对应模型的一个placeholder
    '''
    batch_dict = {}
    batch_dict['question'] = gather_single_column_from_batch(batch_examples, maxlen, 'question', batch_size)
    #batch_dict['qn_entities'] = gather_single_column_from_batch(batch_examples,maxlen, 'qn_entities', batch_size)
    batch_dict['sources'] = gather_single_column_from_batch(batch_examples, maxlen, 'sources', batch_size)
    batch_dict['relations'] = gather_single_column_from_batch(batch_examples, maxlen, 'relations', batch_size)
    batch_dict['targets'] = gather_single_column_from_batch(batch_examples, maxlen, 'targets', batch_size)
    batch_dict['keys'], batch_dict['values'] = gather_key_and_value_from_batch(batch_examples, maxlen, batch_size)
    labels = []
    for i in xrange(batch_size):
        for ans in batch_examples[i]['ans_entities']:
            ans2arr = [0]*entity_size
            ans2arr[ans] = 1
            labels.append(np.array(ans2arr))
    batch_dict['answer'] = np.array(labels)
    return batch_dict

def gather_single_column_from_batch(batch_examples, maxlen, column_name, batch_size):
    '''
    对minibatch数据的某一列进行pad和按照answer个数进行扩展。最终处理完数据的长度会大于batchsize，因为每个例子往往会有好几个答案。
    :param batch_examples: batch数据
    :param maxlen: 各项数据最大长度
    :param column_name: 要处理的列名
    :param batch_size: batchsize大小
    :return: 处理完之后的一列数据，是列表形式保存
    '''
    column = []
    for i in xrange(batch_size):
        num_ans = len(batch_examples[i]['ans_entities'])
        example = pad(batch_examples[i][column_name], maxlen[column_name])
        for j in xrange(num_ans):
            column.append(np.array(example))
    return np.array(column)

def gather_key_and_value_from_batch(batch_examples, maxlen, batch_size):
    '''
    获得数据相关的key和value，其实就是把知识库三元组的主语和关系当做key，把宾语当做value
    :param batch_examples: minibatch的数据
    :param maxlen: 各项的最大长度，其中key和value的最大长度取得是其长度和memory_slot中的最小值
    :param batch_size: 
    :return:  
    '''
    column_key = []
    column_val = []
    for i in xrange(batch_size):
        example_length = len(batch_examples[i]['sources'])
        memories_key = []
        memories_val = []
        src = batch_examples[i]['sources']
        rel = batch_examples[i]['relations']
        tar = batch_examples[i]['targets']
        if maxlen['keys'] > example_length:
            #pad sources, relations and targets in each example
            src = pad(src, maxlen['keys'])
            rel = pad(rel, maxlen['relations'])
            tar = pad(tar, maxlen['targets'])
            example_indices_to_pick = range(len(src))
        else:
            example_indices_to_pick = random.sample(range(example_length), maxlen['keys'])
        
        for memory_index in example_indices_to_pick:
            memories_key.append(np.array([src[memory_index], rel[memory_index]]))
            memories_val.append(tar[memory_index])
        
        num_ans = len(batch_examples[i]['ans_entities'])
        for j in xrange(num_ans):
            column_key.append(np.array(memories_key))
            column_val.append(np.array(memories_val))
    return np.array(column_key), np.array(column_val)