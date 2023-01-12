from torch.utils.data import DataLoader, Dataset
import csv
import json
import copy
import torch
import random
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choice
import utils.common as common
from utils.tokenization import basic_tokenize, convert_to_unicode
from multiprocessing import Pool, cpu_count, Manager
import gensim



class DataProcessor(object):
    """Base class for data converters for token classification data sets."""

    def get_train_examples(self, input_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, input_file):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines


class NerDataProcessor(DataProcessor):
    """Processor for the named entity recognition data set."""

    def __init__(self, args, tokenizer) -> None:
        super(NerDataProcessor, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name
        self.label_file = args.first_label_file
        self.tokenizer = tokenizer
        self.padding_to_max = args.padding_to_max
        self.is_nested = args.exist_nested
        self.span_decode_strategy = args.span_decode_strategy
        self.label_str_file = args.label_str_file

        self.id2label, self.label2id = self.load_labels()
        self.class_num = len(self.id2label)
        self.is_chinese = args.is_chinese

        self.use_random_label_emb = args.use_random_label_emb
        self.use_label_embedding = args.use_label_embedding
        if self.use_label_embedding:
            self.label_ann_word_id_list_file = args.label_ann_word_id_list_file
            args.label_ann_vocab_size = len(self.load_json(args.label_ann_vocab_file))
        self.use_label_encoding = args.use_label_encoding
        self.label_list = args.label_list
        self.token_ids = None
        self.input_mask = None
        self.token_type_ids = None

        # add soft lexicon
        self.gaze_vocab_dict = self.load_json(args.gaz_position_file)
        self.vocab_count_dict = self.load_json(args.vocab_count_file)
        self.word2vec = gensim.models.KeyedVectors.load(args.word_embedding_file, mmap='r')
        self.embedding_dims = 300
        self.scale = np.sqrt(3.0 / self.embedding_dims)
    
    # add soft lexicon
    def gen_lexicon_embedding_for_text(self, token_ids):
        soft_lexicon_embbeddings_list = []
        orig_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        tokens_len = len(orig_tokens) 
        # orig tokens is list type
        token_word_list = []
        for index, token in enumerate(orig_tokens):
            if len(token) > 1:
                # word_piece切词可能的情况
                if token.startswith("##"):
                    token = token[2:]
            token_word_embeddings = {'B':[], 'M':[], 'E':[], 'S':[]}
            token_word = {'B':[], 'M':[], 'E':[], 'S':[]} # for test
            if token in self.gaze_vocab_dict:
                cur_gaze_info = self.gaze_vocab_dict.get(token)
                if 'B' in cur_gaze_info:
                    # begin
                    for j in range(index+1, tokens_len):
                        if j == tokens_len - 1:
                            cur_span = orig_tokens[index:]
                        else:
                            cur_span = orig_tokens[index:j+1]
                        # need to trans char list to str
                        cur_span = "".join(cur_span)
                        if cur_span in cur_gaze_info['B']:
                            token_word_embeddings['B'].append((self.vocab_count_dict[cur_span], 
                                                                self.word2vec[cur_span] * self.vocab_count_dict[cur_span]))
                            token_word['B'].append(cur_span)

                if 'E' in cur_gaze_info:
                    # end
                    for j in range(0, index):
                        if index == tokens_len - 1:
                            cur_span = orig_tokens[j:]
                        else:
                            cur_span = orig_tokens[j:index+1]
                        # need to trans char list to str
                        cur_span = "".join(cur_span)
                        if cur_span in cur_gaze_info['E']:
                            token_word_embeddings['E'].append((self.vocab_count_dict[cur_span], 
                                                                self.word2vec[cur_span] * self.vocab_count_dict[cur_span]))
                            token_word['E'].append(cur_span)
                            
                if 'S' in cur_gaze_info:
                    # sigle
                    cur_span = token
                    token_word_embeddings['S'].append((self.vocab_count_dict[cur_span], 
                                                                self.word2vec[cur_span] * self.vocab_count_dict[cur_span]))
                    token_word['S'].append(cur_span)

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
                                token_word_embeddings['M'].append((self.vocab_count_dict[candi_word], 
                                                                self.word2vec[candi_word] * self.vocab_count_dict[candi_word]))
                                token_word['M'].append(candi_word)
            token_word_list.append(token_word)
            final_embeddings_for_cur_token = []
            for key in token_word_embeddings.keys(): # aggregate (B M E S) word embs of cur token
                if token_word_embeddings[key]:
                    cur_embedding_res = np.sum([ele[1] for ele in token_word_embeddings[key]], axis=0, keepdims=True)
                    total_count = np.sum([ele[0] for ele in token_word_embeddings[key]])
                    cur_embedding_res = 4 * cur_embedding_res / total_count
                else:
                    cur_embedding_res = np.random.uniform(-self.scale, self.scale, [1, self.embedding_dims])
                final_embeddings_for_cur_token.append(cur_embedding_res)
            
            final_embeddings_for_cur_token = np.concatenate(final_embeddings_for_cur_token, axis=-1)
            soft_lexicon_embbeddings_list.append(final_embeddings_for_cur_token)
        soft_lexicon_embbeddings_list = np.concatenate(soft_lexicon_embbeddings_list, axis=0)
        return soft_lexicon_embbeddings_list, token_word_list


    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def load_json(self, input_file):
        with open(input_file, 'r') as fr:
            loaded_data = json.load(fr)
        return loaded_data

    def load_labels(self):
        """See base class."""
        with open(self.label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_class_num(self):
        return self.class_num

    def get_label_data(self, device, rebuild=False):
        if rebuild:
            self.token_ids = None
            self.input_mask = None
            self.token_type_ids = None
        if self.token_ids is None:
            if self.use_label_embedding:
                token_ids, input_mask = [], []
                max_len = 0
                with open(self.label_ann_word_id_list_file, 'r') as fr:
                    for line in fr.readlines():
                        if line != '\n':
                            token_ids.append([int(item) for item in line.strip().split(' ')])
                            max_len = max(max_len, len(token_ids[-1]))
                            input_mask.append([1] * len(token_ids[-1]))
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
            else:
                if self.use_label_encoding:
                    with open(self.label_list, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                else:
                    with open(self.label_str_file, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                token_ids, input_mask, max_len = [], [], 0
                # add label lexicon
                label_lexicon_embeddings_list = []
                for label_str in label_str_list:
                    encoded_results = self.tokenizer.encode_plus(label_str, add_special_tokens=True)
                    token_id = encoded_results['input_ids']
                    input_mask.append(encoded_results['attention_mask'])
                    max_len = max(max_len, len(token_id))
                    token_ids.append(token_id)
                    #
                    label_lexicon_embeddings, token_word_list = self.gen_lexicon_embedding_for_text(token_id)
                    label_lexicon_embeddings_list.append(label_lexicon_embeddings)

                assert max_len <= self.max_seq_length and len(token_ids) == self.class_num
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                    label_lexicon_embeddings_list[idx] = np.pad(label_lexicon_embeddings_list[idx], ((0, padding_length), (0, 0)))
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
                self.token_type_ids = torch.zeros_like(self.token_ids, dtype=torch.long).to(device)
                self.label_lexicon_embeddings_list = torch.tensor(np.array(label_lexicon_embeddings_list), dtype=torch.float32).to(device)
        return self.token_ids, self.token_type_ids, self.input_mask, self.label_lexicon_embeddings_list

    def decode_label4span(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.is_nested:
            assert batch_match_label_ids is not None
            return self._extract_nested_span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)
        elif self.span_decode_strategy == "v5":
            return self._extract_span_v5(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        elif self.span_decode_strategy == "v1":
            return self._extract_span_v1(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
        else:
            raise ValueError("no {} span decoding strategy.".format(self.span_decode_strategy))

    def decode_label4crf(self):
        pass

    def decode_label(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        if self.model_name == "SERS" or self.model_name == "bert_ner":
            return self.decode_label4span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)
        elif "crf" in self.model_name:
            return self.decode_label4crf()

    def encode_labels(self, entities, seq_len, offset_dict, tokens):
        first_label_start_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        first_label_end_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        if self.is_nested:
            match_label = torch.zeros((seq_len, seq_len, self.class_num), dtype=torch.float32)
        golden_labels, entity_cnt = [], 0

        for label in entities:
            label_start_offset = label["start_offset"]
            label_end_offset = label["end_offset"]
            try:
                start_idx = offset_dict[label_start_offset]
                end_idx = offset_dict[label_end_offset]
            except:
                # logger.warn(tokens)
                # logger.warn("{},{},{}".format(
                #     text[label_start_offset:label_end_offset+1], label_start_offset, label_end_offset))
                errmsg = "first_label '{}' doesn't exist in '{}'\n".format(label['text'], ' '.join(tokens))
                common.logger.warn(errmsg)
                continue
            if end_idx >= seq_len:
                continue
            if not self.is_chinese:
                # print("=============== not chinese ===================")
                # print(self.is_chinese)
                # x1 = ''.join(tokens[start_idx:end_idx+1])
                # print(x1)
                # x1_rep = x1.replace("##", "").lower()
                # print(x1_rep)
                # print(label['text'])
                # print(label['text'].lower().replace(" ", ""))
                # print("==================================")
                assert ''.join(tokens[start_idx:end_idx+1]).replace("##",
                                                                    "").lower() == label['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[start_idx:end_idx+1]).replace("##", "").lower(), label['text'].lower().replace(" ", ""))
            entity_cnt += 1
            label_id = self.label2id[label['label']]
            golden_labels.append(common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))

            first_label_start_ids[start_idx][label_id] = 1
            first_label_end_ids[end_idx][label_id] = 1
            if self.is_nested:
                match_label[start_idx, end_idx, label_id] = 1
        results = {
            'entity_starts': first_label_start_ids,
            'entity_ends': first_label_end_ids,
            'entity_cnt': entity_cnt,
            'golden_labels': golden_labels
        }
        if self.is_nested:
            results['match_label'] = match_label
        return results

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line['id'] = guid
            examples.append(line)
        return examples

    @ classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def load_examples(self, input_file, data_type):
        examples = self._create_examples(self._read_json(input_file), data_type)
        return examples

    def build_offset_mapping(self, offsets, tokens):
        offset_dict = {}
        for token_idx in range(len(tokens)):
            # skip [cls] and [sep]
            if token_idx == 0 or token_idx == (len(tokens) - 1):
                continue
            token_start, token_end = offsets[token_idx]
            offset_dict[token_start] = token_idx
            offset_dict[token_end] = token_idx
        return offset_dict

    # 将数据集转换为特征的重要函数，增加soft_lexicon时需要更改
    def convert_examples_to_feature(self, input_file, data_type):
        features = []
        stat_info = {'entity_cnt': 0, 'max_token_len': 0}
        examples = self.load_examples(input_file, data_type)
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        # 对example中每个token求他的词汇信息
        for example_idx, example in enumerate(examples):
            encoded_results = self.tokenizer.encode_plus(example['text'], add_special_tokens=True, return_offsets_mapping=True)
            token_ids = encoded_results['input_ids']
            token_type_ids = encoded_results['token_type_ids']
            input_mask = encoded_results['attention_mask']
            offsets = encoded_results['offset_mapping']
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            offset_dict = self.build_offset_mapping(offsets, tokens)

            stat_info['max_token_len'] = max(len(token_ids)-2, stat_info['max_token_len'])

            token_ids = token_ids[:self.max_seq_length]
            token_type_ids = token_type_ids[:self.max_seq_length]
            input_mask = input_mask[:self.max_seq_length]

            if token_ids[-1] != sep_id:
                assert len(token_ids) == self.max_seq_length
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            results = self.encode_labels(example['entities'], seq_len, offset_dict, tokens)
            stat_info['entity_cnt'] += results['entity_cnt']

            # add soft lexicon for current tokens
            soft_lexicon_embs, token_word_list = self.gen_lexicon_embedding_for_text(token_ids) #(seq_len, embedding_dim)

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)
            # lexicon
            soft_lexicon_embs = torch.tensor(soft_lexicon_embs, dtype=torch.float32)

            assert len(token_ids) == len(input_mask) == len(token_type_ids) == len(results['entity_starts']) == len(results['entity_ends'])

            if example_idx < 5:
                common.logger.info("*** Example ***")
                common.logger.info("guid: %s", example['id'])
                common.logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                common.logger.info("input_ids: %s", " ".join([str(x) for x in token_ids]))
                common.logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                common.logger.info("segment_ids: %s", " ".join([str(x) for x in token_type_ids]))
                common.logger.info("start_ids: %s" % " ".join([str(x) for x in results['entity_starts']]))
                common.logger.info("end_ids: %s" % " ".join([str(x) for x in results['entity_ends']]))
                common.logger.info("%s" % "\n".join([str(x) for x in token_word_list]))
                common.logger.info(f"soft lexicon embeddings shape: {soft_lexicon_embs.shape}")


            features.append(NerFeatures(example_id=example['id'],
                                        tokens_ids=token_ids,
                                        input_mask=input_mask,
                                        seq_len=seq_len,
                                        token_type_ids=token_type_ids,
                                        soft_lexicon_embs=soft_lexicon_embs, # add soft lexicon
                                        first_label_start_ids=results['entity_starts'],
                                        first_label_end_ids=results['entity_ends'],
                                        golden_label=results['golden_labels'],
                                        match_label=results['match_label'] if self.is_nested else None))
        return {'features': features, "stat_info": stat_info}

    def build_seqlens_from_mask(self, input_mask):
        seqlens = [seq_mask.sum() for seq_mask in input_mask]
        return seqlens

    def _extract_span_v5(self, starts, ends, seqlens=None, position_dict=None, scores=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        # print(seqlens)
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)
        if return_span_score:
            assert scores is not None
            span_score_list = [[] for _ in range(starts.shape[0])]
        if seqlens is not None:
            assert starts.shape[0] == len(seqlens)
        if return_cnt:
            span_cnt = 0
        label_num = starts.shape[-1]
        span_list = [[] for _ in range(starts.shape[0])]

        for batch_idx in range(starts.shape[0]):
            for label_idx in range(label_num):

                cur_spans = []

                seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                                   ] if seqlens is not None else starts[batch_idx, :, label_idx]
                seq_end_labels = ends[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else ends[batch_idx, :, label_idx]

                start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
                for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                    if token_start_prob >= s_limit:
                        if end_idx != -1:  # build span
                            if return_span_score:
                                cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                           end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                            else:
                                cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                                  end_idx=end_idx, label_id=label_idx))
                            start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1  # reset state
                        if token_start_prob > start_prob:  # new start, if pre prob is lower, drop it
                            start_prob = token_start_prob
                            start_idx = token_idx
                    if token_end_prob > e_limit and start_prob > s_limit:  # end
                        if token_end_prob > end_prob:
                            end_prob = token_end_prob
                            end_idx = token_idx
                if end_idx != -1:
                    if return_span_score:
                        cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                   end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                    else:
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=end_idx, label_id=label_idx))
                cur_spans = list(set(cur_spans))
                if return_cnt:
                    span_cnt += len(cur_spans)
                if return_span_score:
                    span_score_list[batch_idx].extend(
                        [(item.start_score, item.end_score) for item in cur_spans])
                    span_list[batch_idx].extend([common.LabelSpan(
                        start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
                else:
                    span_list[batch_idx].extend(cur_spans)
        output = (span_list,)
        if return_cnt:
            output += (span_cnt,)
        if return_span_score:
            output += (span_score_list,)
        return output

    def _extract_span_v1(self, starts, ends, seqlens=None, position_dict=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        if seqlens is not None:
            assert starts.shape[0] == seqlens.shape[0]
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)

        label_num = starts.shape[-1]
        span_list = [[] for _ in range(starts.shape[0])]
        for batch_idx in range(starts.shape[0]):
            for label_idx in range(label_num):
                if seqlens is not None:
                    cur_start_idxes = np.where(
                        starts[batch_idx, :seqlens[batch_idx], label_idx] > s_limit)
                    cur_end_idxes = np.where(
                        ends[batch_idx, :seqlens[batch_idx], label_idx] > e_limit)
                else:
                    cur_start_idxes = np.where(
                        starts[batch_idx, :, label_idx] > s_limit)
                    cur_end_idxes = np.where(
                        ends[batch_idx, :, label_idx] > e_limit)

                if cur_start_idxes[0].size == 0 or cur_end_idxes[0].size == 0:
                    continue
                # cur_start_idxes = np.array([pos for pos in cur_start_idxes])
                # cur_end_idxes = np.array([pos[0] for pos in cur_end_idxes])
                cur_start_idxes = cur_start_idxes[0]
                cur_end_idxes = cur_end_idxes[0]
                # print(cur_start_idxes)
                # print(cur_end_idxes)
                if position_dict is not None:
                    cur_start_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                          for idx in cur_start_idxes]))
                    cur_end_idxes = np.unique(np.array([position_dict[batch_idx][idx]
                                                        for idx in cur_end_idxes]))
                cur_spans = []
                # print(cur_start_idxes)
                # print(cur_end_idxes)
                for start_idx in cur_start_idxes:
                    cur_ends = cur_end_idxes[cur_end_idxes >= start_idx]
                    if len(cur_ends) > 0:
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=cur_ends[0], label_id=int(label_idx)))
                cur_spans = list(set(cur_spans))
                span_list[batch_idx].extend(cur_spans)
        return (span_list,)

    def _extract_nested_span(self, starts, ends, seq_lens, matches, is_logits=True, limit=0.5):
        """ for nested"""
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)

        # [batch_size, seq_len]
        batch_size, seq_len, class_num = starts.size()

        # [batch_size, seq_len, class_num]
        extend_input_mask = torch.zeros_like(starts, dtype=torch.long)
        for batch_idx, seq_len_ in enumerate(seq_lens):
            extend_input_mask[batch_idx][:seq_len_] = 1

        start_label_mask = extend_input_mask.unsqueeze(
            -2).expand(-1, -1, seq_len, -1).bool()
        end_label_mask = extend_input_mask.unsqueeze(
            -3).expand(-1, seq_len, -1, -1).bool()

        # [batch_size, seq_len, seq_len, class_num]
        match_infer = matches > 0 if is_logits else matches > limit
        # [batch_size, seq_len, class_num]
        start_infer = starts > 0 if is_logits else starts > limit
        # [batch_size, seq_len, class_num]
        end_infer = ends > 0 if is_logits else ends > limit

        start_infer = start_infer.bool()
        end_infer = end_infer.bool()

        # match_infer = torch.ones_like(match_infer)

        match_infer = (
            match_infer & start_infer.unsqueeze(2).expand(-1, -1, seq_len, -1)
            & end_infer.unsqueeze(1).expand(-1, seq_len, -1, -1))

        match_label_mask = torch.triu((start_label_mask & end_label_mask).permute(
            0, 3, 1, 2).contiguous().view(-1, seq_len, seq_len), 0).contiguous().view(
            batch_size, class_num, seq_len, seq_len).permute(0, 2, 3, 1)

        # [batch_size, seq_len, seq_len, class_num]
        match_infer = match_infer & match_label_mask

        span_list = [[] for _ in range(batch_size)]
        items = torch.where(match_infer == True)
        if len(items[0]) != 0:
            for idx in range(len(items[0])):
                batch_idx = int(items[0][idx])
                start_idx = int(items[1][idx])
                end_idx = int(items[2][idx])
                label_id = int(items[3][idx])
                span_list[batch_idx].append(
                    common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))
        return (span_list,)

    def build_output_results(self, tokens, infers, goldens=None):
        outputs = []
        for batch_idx, (token, seq_infers) in enumerate(zip(tokens, infers)):
            text = self.tokenizer.decode(token, skip_special_tokens=True)
            infer_list = [{'text': self.tokenizer.decode(token[infer.start_idx:infer.end_idx+1]),
                           'label':self.id2label[infer.label_id]} for infer in seq_infers]
            outputs.append({
                'text': text,
                'entity_infers': infer_list
            })
            if goldens is not None:
                join_set = set(goldens[batch_idx]) & set(seq_infers)
                lack = set(goldens[batch_idx]) - join_set
                new = set(seq_infers) - join_set
                outputs[-1]['entity_goldens'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                                  'label':self.id2label[item.label_id]} for item in goldens[batch_idx]]
                outputs[-1]['lack'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                        'label':self.id2label[item.label_id]} for item in lack]
                outputs[-1]['new'] = [{'text': self.tokenizer.decode(token[item.start_idx:item.end_idx+1]),
                                       'label':self.id2label[item.label_id]} for item in new]
        return outputs

    def _generate_batch(self, batch):
        batch_size, class_num = len(
            batch), batch[0].first_label_start_ids.shape[-1]

        batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
        ids = [f.example_id for f in batch]
        batch_golden_label = [f.golden_label for f in batch]
        max_len = int(max(batch_seq_len))

        batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
            (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

        batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
            (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)
        if self.is_nested:
            batch_match_label = torch.zeros(
                (batch_size, max_len, max_len, class_num), dtype=torch.float32)
        
        # add soft lexicon
        batch_soft_lexicon_embs_list = []
        for batch_idx in range(batch_size):
            batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                        ] = batch[batch_idx].tokens_ids
            batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                            ] = batch[batch_idx].token_type_ids
            batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                        ] = batch[batch_idx].input_mask
            batch_first_label_start_ids[batch_idx][:
                                                   batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
            batch_first_label_end_ids[batch_idx][:
                                                 batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids
            if self.is_nested:
                batch_match_label[batch_idx, :batch[batch_idx].match_label.shape[0],
                                  :batch[batch_idx].match_label.shape[1]] = batch[batch_idx].match_label
            soft_lexicon_embeddings = batch[batch_idx].soft_lexicon_embs
            soft_lexicon_embeddings = np.pad(soft_lexicon_embeddings, ((0, max_len - len(soft_lexicon_embeddings)), (0, 0)))
            batch_soft_lexicon_embs_list.append(soft_lexicon_embeddings)

        batch_soft_lexicon_embs = torch.tensor(np.array(batch_soft_lexicon_embs_list), dtype=torch.float32)

        results = {'token_ids': batch_tokens_ids,
                   'token_type_ids': batch_token_type_ids,
                   'input_mask': batch_input_mask,
                   'seq_len': batch_seq_len,
                   'first_starts': batch_first_label_start_ids,
                   'first_ends': batch_first_label_end_ids,
                   'ids': ids,
                   'golden_label': batch_golden_label,
                   'soft_lexicon_embs': batch_soft_lexicon_embs
                   }
        if self.is_nested:
            results['match_label'] = batch_match_label
        return results

    def generate_batch_data(self):
        return self._generate_batch


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids, ori_tokens=None, first_label_start_ids=None, first_label_end_ids=None,
                 first_label_start=None,
                 first_label_end=None,
                 second_label_start_ids=None,
                 second_label_end_ids=None, golden_label=None,  text_type_ids=None, relative_pos_label=None,
                 extra_mask=None, second_label_mask=1.0, first_label_ids=None, extended_seq_len=None, scores_ids=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.first_label_start = first_label_start
        self.first_label_end = first_label_end
        self.second_label_start_ids = second_label_start_ids
        self.second_label_end_ids = second_label_end_ids
        self.golden_label = golden_label
        self.second_label_mask = second_label_mask
        self.ori_tokens = ori_tokens
        self.text_type_ids = text_type_ids
        self.relative_pos_label = relative_pos_label
        self.extra_mask = extra_mask
        self.first_label_ids = first_label_ids
        self.extended_seq_len = extended_seq_len
        self.scores_ids = scores_ids
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NerFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids, soft_lexicon_embs = None, first_label_start_ids=None, first_label_end_ids=None,
                 golden_label=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.golden_label = golden_label
        self.match_label = match_label
        # add soft lexicon
        self.soft_lexicon_embs = soft_lexicon_embs

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TriggerFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids, first_label_start_ids=None, first_label_end_ids=None,
                 golden_label=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.golden_label = golden_label
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def collate_fn_relation(batch):
    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]

    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_start_ids = torch.tensor([f.first_label_start_ids for f in batch], dtype=torch.float32)[:,
                                                                                                              :max_len]
    batch_first_label_end_ids = torch.tensor(
        [f.first_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_start = torch.tensor(
        [f.first_label_start for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_end = torch.tensor(
        [f.first_label_end for f in batch], dtype=torch.float32)[:, :max_len]
    batch_second_label_start_ids = torch.tensor([f.second_label_start_ids for f in batch], dtype=torch.float32, )[:,
                                                                                                                  :max_len]
    batch_second_label_end_ids = torch.tensor(
        [f.second_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    gold_label = [f.golden_label for f in batch]
    batch_second_label_mask = torch.tensor(
        [f.second_label_mask for f in batch], dtype=torch.float32)
    # batch_relative_pos_ids = torch.tensor(
    #     [f.relative_pos_label for f in batch], dtype=torch.long)[:, :max_len]
    batch_ori_tokens = [f.ori_tokens for f in batch]
    batch_extra_mask = torch.tensor(
        [f.extra_mask for f in batch], dtype=torch.long)[:, :max_len]
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'first_start': batch_first_label_start,
            'first_end': batch_first_label_end,
            'second_starts': batch_second_label_start_ids,
            'second_ends': batch_second_label_end_ids,
            'second_label_mask': batch_second_label_mask,
            # 'relative_pos_ids': batch_relative_pos_ids,
            'golden_label': gold_label,
            'ori_tokens': batch_ori_tokens,
            'extra_mask': batch_extra_mask,
            'ids': ids}


def collate_fn_event(batch):
    batch_size = len(batch)

    event_str_ids = np.load(
        '/home/yangpan/workspace/onepass_ie/data/ace05/splited/event_type_uncased_whole_ids.npy', allow_pickle=True)
    # print(event_str_ids.shape)
    batch_event_str_ids = torch.tensor(
        np.repeat(event_str_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
    # print(batch_event_str_ids.shape)

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    batch_extended_seq_len = [f.extended_seq_len for f in batch]
    # max_len = int(max(batch_extended_seq_len))
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]

    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_start_ids = torch.tensor([f.first_label_start_ids for f in batch], dtype=torch.float32)[:,
                                                                                                              :max_len]
    batch_first_label_end_ids = torch.tensor(
        [f.first_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_start = torch.tensor(
        [f.first_label_start for f in batch], dtype=torch.float32)[:, :max_len]
    batch_first_label_end = torch.tensor(
        [f.first_label_end for f in batch], dtype=torch.float32)[:, :max_len]
    batch_second_label_start_ids = torch.tensor([f.second_label_start_ids for f in batch], dtype=torch.float32, )[:,
                                                                                                                  :max_len]
    batch_second_label_end_ids = torch.tensor(
        [f.second_label_end_ids for f in batch], dtype=torch.float32)[:, :max_len]
    gold_label = [f.golden_label for f in batch]
    batch_second_label_mask = torch.tensor(
        [f.second_label_mask for f in batch], dtype=torch.float32)
    batch_text_type_ids = torch.tensor(
        [f.text_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_ori_tokens = [f.ori_tokens for f in batch]
    batch_extra_mask = torch.tensor(
        [f.extra_mask for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_ids = torch.tensor(
        [f.first_label_ids for f in batch], dtype=torch.float32)
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'first_start': batch_first_label_start,
            'first_end': batch_first_label_end,
            'second_starts': batch_second_label_start_ids,
            'second_ends': batch_second_label_end_ids,
            'second_label_mask': batch_second_label_mask,
            'text_type_ids': batch_text_type_ids,
            'golden_label': gold_label,
            'ids': ids,
            'first_types': batch_first_label_ids,
            'extra_mask': batch_extra_mask,
            'ori_tokens': batch_ori_tokens, 'event_str_ids': batch_event_str_ids}
    #
    # batch_tokens_ids, batch_input_mask, batch_seq_len, batch_token_type_ids, batch_sub_head_ids, batch_sub_end_ids, batch_sub_head, batch_sub_end, \
    # batch_obj_head_ids, batch_obj_end_ids = map(torch.stack, zip(*batch))
    #
    # max_len = int(max(batch_seq_len).item())
    # # print(batch_seq_len)
    # batch_tokens_ids = batch_tokens_ids[:, :max_len]
    # batch_token_type_ids = batch_token_type_ids[:, :max_len]
    # batch_input_mask = batch_input_mask[:, :max_len]
    # # batch_seq_len = batch_seq_len[:, :max_len]
    # batch_sub_head_ids = batch_sub_head_ids[:, :max_len]
    # batch_sub_end_ids = batch_sub_end_ids[:, :max_len]
    # batch_sub_head = batch_sub_head[:, :max_len]
    # batch_sub_end = batch_sub_end[:, :max_len]
    # batch_obj_head_ids = batch_obj_head_ids[:, :max_len]
    # batch_obj_end_ids = batch_obj_end_ids[:, :max_len]
    # return {'token_ids': batch_tokens_ids,
    #         'token_type_ids': batch_token_type_ids,
    #         'input_mask': batch_input_mask,
    #         'seq_len': batch_seq_len,
    #         'first_heads': batch_sub_head_ids,
    #         'first_tails': batch_sub_end_ids,
    #         'sub_head': batch_sub_head,
    #         'sub_tail': batch_sub_end,
    #         'second_heads': batch_obj_head_ids,
    #         'second_tails': batch_obj_end_ids}


def collate_fn_classification(batch):
    batch_size = len(batch)
    # label_ids = np.load(
    #     '/data/yangpan/workspace/dataset/text_classification/imdb/emotional_orientation_uncased_whole_ids.npy', allow_pickle=True)

    # batch_label_ids = torch.tensor(
    #     np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    # batch_extended_seq_len = [f.extended_seq_len for f in batch]
    max_len = int(max(batch_seq_len).item())
    batch_tokens_ids = torch.tensor(
        [f.tokens_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_input_mask = torch.tensor(
        [f.input_mask for f in batch], dtype=torch.float32)[:, :max_len]
    batch_token_type_ids = torch.tensor(
        [f.token_type_ids for f in batch], dtype=torch.long)[:, :max_len]
    batch_first_label_ids = torch.tensor(
        [f.first_label_ids for f in batch], dtype=torch.long)
    # batch_ori_tokens = [f.ori_tokens for f in batch]
    ids = [f.example_id for f in batch]
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'ids': ids,
            'first_types': batch_first_label_ids,
            # 'first_label_ids': batch_label_ids,
            # 'ori_tokens': batch_ori_tokens
            }


def collate_fn_multi_classification(batch):
    batch_size = len(batch)

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    # batch_label_ids = torch.tensor(
    #     [f.label_ids for f in batch], dtype=torch.float32)

    batch_label_ids = torch.stack([f.label_ids for f in batch], dim=0)

    # print(batch_label_ids.shape)
    # print(batch_tokens_ids)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        # print(batch[batch_idx].tokens_ids)
        # print(batch_tokens_ids[batch_idx])
        # print(batch_label_ids[batch_idx])
        # print('\n')
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_types': batch_label_ids,
            'ids': ids
            }


def collate_fn_trigger(batch):
    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
        (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)

    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
        batch_first_label_end_ids[batch_idx][:
                                             batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            }


def collate_fn_trigger_crf(batch):
    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids = torch.zeros(
        (batch_size, max_len), dtype=torch.long)

    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            }


def collate_fn_ner_crf(batch):

    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids = torch.zeros(
        (batch_size, max_len), dtype=torch.long)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'ids': ids,
            'golden_label': batch_golden_label
            }


def collate_fn_ner(batch):

    batch_size, class_num = len(
        batch), batch[0].first_label_start_ids.shape[-1]

    batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
    ids = [f.example_id for f in batch]
    batch_golden_label = [f.golden_label for f in batch]
    max_len = int(max(batch_seq_len))

    batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
        (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

    batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
        (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)
    batch_match_label = torch.zeros(
        (batch_size, max_len, max_len, class_num), dtype=torch.float32)
    for batch_idx in range(batch_size):
        batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                    ] = batch[batch_idx].tokens_ids
        batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                        ] = batch[batch_idx].token_type_ids
        batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                    ] = batch[batch_idx].input_mask
        batch_first_label_start_ids[batch_idx][:
                                               batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
        batch_first_label_end_ids[batch_idx][:
                                             batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids
        batch_match_label[batch_idx, :batch[batch_idx].match_label.shape[0],
                          :batch[batch_idx].match_label.shape[1]] = batch[batch_idx].match_label

    return {'token_ids': batch_tokens_ids,
            'token_type_ids': batch_token_type_ids,
            'input_mask': batch_input_mask,
            'seq_len': batch_seq_len,
            'first_starts': batch_first_label_start_ids,
            'first_ends': batch_first_label_end_ids,
            'ids': ids,
            'golden_label': batch_golden_label,
            'match_label': batch_match_label
            }

# def collate_fn_ner(batch):
#     batch_size = len(batch)

#     # label_ids = np.load(
#     #     '/data/yangpan/workspace/dataset/ner/conll03/processed/entity_with_annoation_uncased_whole_ids.npy', allow_pickle=True)

#     # batch_label_ids = torch.tensor(
#     #     np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)

#     batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
#     max_len = int(max(batch_seq_len))
#     batch_tokens_ids = torch.tensor(
#         [f.tokens_ids for f in batch], dtype=torch.long)
#     batch_input_mask = torch.tensor(
#         [f.input_mask for f in batch], dtype=torch.float32)

#     batch_token_type_ids = torch.tensor(
#         [f.token_type_ids for f in batch], dtype=torch.long)

#     batch_first_label_start_ids = torch.tensor(
#         [f.first_label_start_ids for f in batch], dtype=torch.float32)
#     batch_first_label_end_ids = torch.tensor(
#         [f.first_label_end_ids for f in batch], dtype=torch.float32)
#     # batch_first_label_ids = torch.tensor(
#     #     [f.first_label_ids for f in batch], dtype=torch.float32)
#     batch_match_label = torch.tensor(
#         [f.match_label for f in batch], dtype=torch.float32)

#     # batch_ori_tokens = [f.ori_tokens for f in batch]
#     ids = [f.example_id for f in batch]

#     # batch_scores_ids = torch.tensor(
#     #     [f.scores_ids for f in batch], dtype=torch.float32)[:, : max_len]

#     label_num = batch_first_label_end_ids.shape[-1]
#     label_ids = np.array([idx for idx in range(label_num)], dtype=np.int)
#     batch_label_ids = torch.tensor(
#         np.repeat(label_ids[np.newaxis, :], batch_size, axis=0), dtype=torch.long)
#     batch_golden_label = [f.golden_label for f in batch]

#     return {'token_ids': batch_tokens_ids,
#             'token_type_ids': batch_token_type_ids,
#             'input_mask': batch_input_mask,
#             'seq_len': batch_seq_len,
#             'first_starts': batch_first_label_start_ids,
#             'first_ends': batch_first_label_end_ids,
#             'first_label_ids': batch_label_ids,
#             'ids': ids,
#             # 'first_types': batch_first_label_ids,
#             # 'ori_tokens': batch_ori_tokens,
#             # 'scores_ids': batch_scores_ids,
#             'golden_label': batch_golden_label,
#             'match_label': batch_match_label}


class DataPreFetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].to(device=self.device,
                                                             non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
