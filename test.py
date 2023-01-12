import os
import shutil
from bertviz import head_view
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
# from models.model_ner import BertNerSpan, BertNerSpanMatrix, BertNerSpanMatrixWithLabelSiamese, BertNerSpanWithLabel, BertNerSpanWithLabelSiamese
import models.model_ner as models
import torch
import numpy as np
from utils.losses import *
from transformers import AdamW, BertTokenizerFast
from torch.optim import lr_scheduler
from callback.lr_scheduler import get_linear_schedule_with_warmup
from utils.common import EntityLabelWithScore, seed_everything, init_logger, logger, load_model, EntityLabel
import json
import time
from data_loader import NerDataProcessor
from utils.finetuning_args import get_argparse, print_arguments
from utils.evaluate import MetricsCalculator4Ner
import multiprocessing
import pickle
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
import gensim
from tqdm import tqdm
from utils.common import init_logger, logger


gaz_vocab_file = "./data/words/zh_onto4/gaz_position_dict.json"
vocab_count_file = "./data/words/zh_onto4/soft_vocab_count.json"
word_embedding_file = "./data/words/zh_onto4/bio_word2vec_trim"

def load_json_file(file):
    with open(file, "r", encoding='utf-8') as fr:
        result = json.load(fr)
    return result

gaze_vocab_dict = load_json_file(gaz_vocab_file)
vocab_count_dict = load_json_file(vocab_count_file)
word2vec = gensim.models.KeyedVectors.load(word_embedding_file, mmap='r')

# logger.info(f"gaze_vocab_dict len: {len(gaz_vocab_file)}")
# logger.info(f"vocab_count_dict len: {len(vocab_count_dict)}")
bert_model_path = "/home/ybb/Project/wt/bert_pretrained/chinese_roberta_wwm_large_ext_pytorch"

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=False)

embedding_dims = 300
scale = np.sqrt(3.0 / embedding_dims)

def gen_lexicon_embedding_for_text(token_ids):
        soft_lexicon_embbeddings_list = []
        orig_tokens = tokenizer.convert_ids_to_tokens(token_ids)

        tokens_len = len(orig_tokens)
        logger.info(f"orig tokens: {orig_tokens}")
        logger.info(f"tokens len: {tokens_len}")
        logger.info("===========================")
        for index, token in tqdm(enumerate(orig_tokens)):
            if len(token) > 1:
                # word_piece切词可能的情况
                if token.startswith("##"):
                    token = token[2:]
            token_word_embeddings = {'B':[], 'M':[], 'E':[], 'S':[]}
            token_word = {'B':[], 'M':[], 'E':[], 'S':[]}
            if token in gaze_vocab_dict:
                cur_gaze_info = gaze_vocab_dict.get(token)
                logger.info(f"gaz info : {cur_gaze_info}")
                if 'B' in cur_gaze_info:
                    # begin
                    for j in range(index+1, tokens_len):
                        if j == tokens_len - 1:
                            cur_span = orig_tokens[index:]
                        else:
                            cur_span = orig_tokens[index:j+1]
                        cur_span = "".join(cur_span)
                        if cur_span in cur_gaze_info['B']:
                            token_word_embeddings['B'].append((vocab_count_dict[cur_span], 
                                                                word2vec[cur_span] * vocab_count_dict[cur_span]))
                            # logger.info(f"B - cur_span: {cur_span}  embedding shape: {word2vec[cur_span].shape}")
                            token_word["B"].append(cur_span)
                if 'E' in cur_gaze_info:
                    # end
                    for j in range(0, index):
                        if index == tokens_len - 1:
                            cur_span = orig_tokens[j:]
                        else:
                            cur_span = orig_tokens[j:index+1]
                        cur_span = "".join(cur_span)
                        if cur_span in cur_gaze_info['E']:
                            token_word_embeddings['E'].append((vocab_count_dict[cur_span], 
                                                                word2vec[cur_span] * vocab_count_dict[cur_span]))
                            # logger.info(f"E - cur_span: {cur_span}  embedding shape: {word2vec[cur_span].shape}")
                            token_word["E"].append(cur_span)

                if 'S' in cur_gaze_info:
                    # sigle
                    cur_span = token
                    token_word_embeddings['S'].append((vocab_count_dict[cur_span], 
                                                                word2vec[cur_span] * vocab_count_dict[cur_span]))
                    # logger.info(f"S - cur_span: {cur_span}  embedding shape: {word2vec[cur_span].shape}")
                    token_word["S"].append(cur_span)

                if 'M' in cur_gaze_info:
                    # mid
                    if index > 0 and index < (tokens_len - 1):
                        candi_word_list = cur_gaze_info['M']
                        for candi_word in candi_word_list:
                            word_len = len(candi_word)
                            char_list = [ch for ch in candi_word]
                            char_index = char_list.index(token)
                            if char_index > index or index - char_index + word_len > tokens_len:
                                continue
                            pick_span = orig_tokens[index - char_index : index - char_index + word_len]
                            pick_span = "".join(pick_span)
                            if candi_word == pick_span:
                                token_word_embeddings['M'].append((vocab_count_dict[candi_word], 
                                                                word2vec[candi_word] * vocab_count_dict[candi_word]))
                                # logger.info(f"M - cur_span: {candi_word}  embedding shape: {word2vec[candi_word].shape}")
                                token_word['M'].append(candi_word)
            final_embeddings_for_cur_token = []
            logger.info(f"current token words: {token_word}")
            for key in token_word_embeddings.keys(): # token for B M E S
                if token_word_embeddings[key]:
                    cur_embedding_res = np.sum([ele[1] for ele in token_word_embeddings[key]], axis=0, keepdims=True)
                    total_count = np.sum([ele[0] for ele in token_word_embeddings[key]])
                    cur_embedding_res = 4 * cur_embedding_res / total_count
                    # print("=====================")
                    # print(cur_embedding_res.shape)
                    # logger.info(f"cur_embedding shape: {cur_embedding_res.shape}")
                else:
                    cur_embedding_res = np.random.uniform(-scale, scale, [1, embedding_dims])
                final_embeddings_for_cur_token.append(cur_embedding_res)
            
            final_embeddings_for_cur_token = np.concatenate(final_embeddings_for_cur_token, axis=-1)
            soft_lexicon_embbeddings_list.append(final_embeddings_for_cur_token)
        soft_lexicon_embbeddings_list = np.concatenate(soft_lexicon_embbeddings_list, axis=0)
        return soft_lexicon_embbeddings_list



if __name__ == "__main__":
    label_file = "./data/ner/zh_onto4/processed/label_annotation.txt"
    log_file = "./test.log"
    init_logger(log_file)
    with open(label_file, "r", encoding="utf-8") as fr:
        label_text_list = [line.strip() for line in fr.readlines()]
    
    label_lexicon_list = []
    max_len = 0
    for label_str in label_text_list:
        encoded_results = tokenizer.encode_plus(label_str, add_special_tokens=True)
        token_ids = encoded_results['input_ids']
        max_len = max(max_len, len(token_ids))
        label_lexicon = gen_lexicon_embedding_for_text(token_ids)
        token = tokenizer.convert_ids_to_tokens(token_ids)
        # logger.info(f"tokens: {token}")
        # logger.info(f"label lexicon embedding: {label_lexicon.shape}")

        label_lexicon_list.append(label_lexicon)
    for idx in range(len(label_lexicon_list)):
        padding_length = max_len - len(label_lexicon_list[idx])
        label_lexicon_list[idx] = np.pad(label_lexicon_list[idx], ((0, padding_length), (0, 0)))
        # logger.info(f"padding length: {padding_length}, label lexicon shape: {label_lexicon_list[idx].shape}")
    label_lexicon_list = torch.tensor(np.array(label_lexicon_list), dtype=torch.float32)
    print(label_lexicon_list[0].shape)
    print(len(label_lexicon_list))

    