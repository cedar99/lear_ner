# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import json
import copy
import torch
import os
from tqdm import tqdm
from gensim.models import KeyedVectors


def get_word_tf(embedding_file, training_context_datas):
    """
    simple lexicon 对出现在语料中词向量词表中的词，计算样本中每个字符分别在B,M,E,S中的向量表示
    先统计词频和出现的词表，（全量，包括测试集合训练集）
    :param embedding_file:
    :return:
    """
    vector_base = KeyedVectors.load_word2vec_format(embedding_file, unicode_errors='ignore')
    # vector_base = KeyedVectors.load(embedding_file, mmap='r')
    print("load word2vec done...")
    embedding_vocab_list = vector_base.wv.index2word
    vocab_count_dict = {key: 0 for key in embedding_vocab_list}
    # {char+B:[word1,word2,...]}
    gaz_position_dict = {}

    text_str = "\n".join(training_context_datas)

    #将匹配数量小于5的vocab添加进来
    words_trim_list = []

    for index, vocab_local in tqdm(enumerate(vocab_count_dict)):
        # if index % 500 == 0 and index > 0:
        #     print(index)
            # break
        try:
            match_count = len(re.findall(re.escape(vocab_local), text_str))
            if match_count < 5:  # 匹配数量少于5
                words_trim_list.append(vocab_local)
                continue
            vocab_count_dict[vocab_local] += match_count
            for index, char in enumerate(vocab_local):
                if len(vocab_local) == 1:
                    # single
                    if char in gaz_position_dict:
                        if "S" not in gaz_position_dict[char]:
                            gaz_position_dict[char]["S"] = [vocab_local]
                        else:
                            gaz_position_dict[char]["S"].append(vocab_local)
                    else:
                        gaz_position_dict[char] = {"S": [vocab_local]}
                        # gaz_position_dict[char]["S"] = [vocab_local]
                else:
                    if index == 0:

                        # start
                        if char in gaz_position_dict:
                            if "B" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["B"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["B"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"B": [vocab_local]}
                    elif index < len(vocab_local) - 1:
                        # middle
                        if char in gaz_position_dict:
                            if "M" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["M"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["M"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"M": [vocab_local]}
                    else:
                        # end
                        if char in gaz_position_dict:
                            if "E" not in gaz_position_dict[char]:
                                gaz_position_dict[char]["E"] = [vocab_local]
                            else:
                                gaz_position_dict[char]["E"].append(vocab_local)
                        else:
                            gaz_position_dict[char] = {"E": [vocab_local]}
        except Exception as e:
            print(e)
            continue

    restrict_w2v(vector_base,words_trim_list)
    # print(len(words_trim_list))
    return vocab_count_dict, gaz_position_dict


def restrict_w2v(vector_base, restricted_word_set):
    """
    根据词表，将频率较高的词取出来，减小embedding的规模
    :param vector_base:
    :param restricted_word_set:
    :return:
    """
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    # new_vectors_norm = []

    for i in tqdm(range(len(vector_base.vocab))):
        word = vector_base.index2entity[i]
        vec = vector_base.vectors[i]
        vocab = vector_base.vocab[word]
        # vec_norm = vector_base.vectors_norm[i]
        if word not in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            # new_vectors_norm.append(vec_norm)

    vector_base.vocab = new_vocab
    vector_base.vectors = np.array(new_vectors)
    vector_base.index2entity = new_index2entity
    vector_base.index2word = new_index2entity
    # vector_base.vectors_norm = new_vectors_norm
    vector_base.init_sims(replace=True)
    # vector_base.save('bio_word2vec')
    vector_base.save('./data/words/zh_msra/bio_word2vec_trim')


def read_json(input_file):
    lines = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(json.loads(line.strip())) 
    return lines

def get_text(input_file):
    line_list = read_json(input_file)
    text_list = []
    for line in line_list:
        text_line = line['text'].replace(" ", "")
        text_list.append(text_line)
    return text_list

if __name__ == "__main__":
    # word2vec = KeyedVectors.load_word2vec_format("data/sgns.financial.bigram-char", binary=False,unicode_errors='ignore')
    # word2vec.init_sims(replace=True)
    train_file = "./data/ner/zh_msra/processed/train.json"
    dev_file = "./data/ner/zh_msra/processed/dev.json"
    test_file = "./data/ner/zh_msra/processed/test.json"
    embedding_dile = "./data/words/sgns.financial.bigram-char"  # 中文词向量文件

    # text_list = read_json(input_file)
    text_list = get_text(train_file)
    text_list.extend(get_text(dev_file))
    text_list.extend(get_text(test_file))

    # print(len(train_list), len(dev_list), len(test_list), len(train_list) + len(dev_list) + len(test_list))

    # print(text_list[:5])


    # max_len = 0
    # for text in text_list:
    #     max_len = max(max_len, len(text))
    # print(max_len)
    root_path = "./data/words/zh_msra"
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    vocab_count_dict, gaz_position_dict = get_word_tf(embedding_dile, text_list)
    print(len(vocab_count_dict))
    print(len(gaz_position_dict))

    with open("./data/words/zh_msra/soft_vocab_count.json", "w") as fw:
        json.dump(vocab_count_dict, fw, ensure_ascii=False, indent=4)
    with open("./data/words/zh_msra/gaz_position_dict.json", "w") as fw:
        json.dump(gaz_position_dict, fw, ensure_ascii=False, indent=4)


