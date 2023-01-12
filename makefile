task_type := sequence_classification
task_save_name := SERS
span_decode_strategy := v5

data_dir := ./data/cache
data_name := zh_msra

model_name := SERS
model_name_or_path := /home/ybb/Project/wt/bert_pretrained/chinese_roberta_wwm_large_ext_pytorch

output_dir := ./zh_msra_models/bert_large
do_lower_case := False

result_dir := ./zh_msra_models/results
first_label_file := ./data/ner/zh_msra/processed/label_map.json
overwrite_output_dir := True

train_set := ./data/ner/zh_msra/processed/train.json
dev_set := ./data/ner/zh_msra/processed/dev.json
test_set := ./data/ner/zh_msra/processed/test.json

is_chinese := True
max_seq_len := 128
label_str_file := ./data/ner/zh_msra/processed/label_annotation.txt
num_epochs := 20
batch_size := 16


##########
use_attn := True
do_add := True

# add soft lexicon
gaz_vocab_file = "./data/words/zh_msra/gaz_position_dict.json"
vocab_count_file = "./data/words/zh_msra/soft_vocab_count.json"
word_embedding_file = "./data/words/zh_msra/bio_word2vec_trim"
add_soft := True


env:
	PYTHONPATH="./:${PYTHONPATH}" \
	CUDA_VISIBLE_DEVICES=1 \

run_ms:
	@python run_ner.py --num_train_epochs ${num_epochs} --is_chinese ${is_chinese} --max_seq_length ${max_seq_len} --task_type ${task_type} --span_decode_strategy ${span_decode_strategy} --task_save_name ${task_save_name} --data_dir ${data_dir} --data_name ${data_name} --model_name ${model_name} --model_name_or_path ${model_name_or_path} --output_dir ${output_dir} --do_lower_case ${do_lower_case} --result_dir ${result_dir} --first_label_file ${first_label_file} --overwrite_output_dir ${overwrite_output_dir} --train_set ${train_set} --dev_set ${dev_set} --test_set ${test_set} --label_str_file ${label_str_file} --use_attn ${use_attn} --per_gpu_train_batch_size ${batch_size} --gaz_position_file ${gaz_vocab_file} --vocab_count_file ${vocab_count_file} --word_embedding_file ${word_embedding_file} --add_soft_lexicon ${add_soft}


run_on:
	@python run_ner.py --task_type sequence_classification --task_save_name SERS --data_dir ${data_dir} --data_name zh_onto4 --model_name SERS --model_name_or_path ${model_name_or_path} --output_dir ./zh_onto4_models/bert_large --do_lower_case False --result_dir ./zh_onto4_models/results --first_label_file ./data/ner/zh_onto4/processed/label_map.json --overwrite_output_dir True --train_set ./data/ner/zh_onto4/processed/train.json --dev_set ./data/ner/zh_onto4/processed/dev.json --test_set ./data/ner/zh_onto4/processed/test.json  --is_chinese True   --max_seq_length 128 --per_gpu_train_batch_size ${batch_size} --gradient_accumulation_steps 1  --num_train_epochs 20 --learning_rate 8e-6  --task_layer_lr 8e-5  --label_str_file ./data/ner/zh_onto4/processed/label_annotation.txt --span_decode_strategy v5 --use_attn ${use_attn} --gaz_position_file ${gaz_vocab_file} --vocab_count_file ${vocab_count_file} --word_embedding_file ${word_embedding_file} --add_soft_lexicon ${add_soft}