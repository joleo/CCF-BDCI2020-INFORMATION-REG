export CUDA_VISIBLE_DEVICES=0,1
CURRENT_DIR=`pwd`
export RUN_PATH=$CURRENT_DIR/src/pytorch_version
python $RUN_PATH/data_process.py

export BERT_BASE_DIR=./data/user_data/language_model/bert_base
export CLUE_DIR=./data/user_data/5fold_mix
export OUTPUT_DIR=./data/user_data/models/outputs_5fold_mix_span_bert2
TASK_NAME='comp'
for i in {0..4}
do
python ./src/pytorch_versiontask_name/run_ner_span_adv.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_predict \
--do_adv \
--do_lower_case \
--loss_type=ce \
--data_dir=$CLUE_DIR/fold_${i}/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=24 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--logging_steps=448 \
--save_steps=448 \
--warmup_steps=448 \
--output_dir=$OUTPUT_DIR/fold_${i}/ \
--overwrite_output_dir \
--overwrite_cache \
--seed=42
done

export BERT_BASE_DIR=./data/user_data/language_model/macbert
export CLUE_DIR=./data/user_data/5fold_mix
export OUTPUT_DIR=./data/user_data/models/outputs_5fold_mix_span_macbert
TASK_NAME='comp'
for i in {0..4}
do
python ./src/pytorch_version/run_ner_span_adv.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_predict \
--do_adv \
--do_lower_case \
--loss_type=ce \
--data_dir=$CLUE_DIR/fold_${i}/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=24 \
--learning_rate=2e-5 \
--num_train_epochs=10 \
--logging_steps=448 \
--save_steps=448 \
--warmup_steps=448 \
--output_dir=$OUTPUT_DIR/fold_${i}/ \
--overwrite_output_dir \
--overwrite_cache \
--seed=42
done

export BERT_BASE_DIR=./data/user_data/language_model/chinese_roberta_wwm_large
export CLUE_DIR=./data/user_data/5fold_mix
export OUTPUT_DIR=./data/user_data/models/outputs_5fold_mix_crf_robert
TASK_NAME='comp'
for i in {0..4}
do
python ./src/pytorch_version/run_ner_crf5.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_predict \
--do_adv \
--do_lower_case \
--data_dir=$CLUE_DIR/fold_${i}/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=24 \
--learning_rate=5e-5 \
--crf_learning_rate=1e-3 \
--num_train_epochs=10 \
--logging_steps=448 \
--save_steps=448 \
--output_dir=$OUTPUT_DIR/fold_${i}/ \
--overwrite_output_dir \
--overwrite_cache \
--seed=42
done

export BERT_BASE_DIR=./data/user_data/language_model/macbert
export CLUE_DIR=./data/user_data/5fold_mix
export OUTPUT_DIR=./data/user_data/models/outputs_5fold_mix_crf_macbert
TASK_NAME='comp'
for i in {0..4}
do
python ./src/pytorch_version/run_ner_crf5.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_predict \
--do_adv \
--do_lower_case \
--data_dir=$CLUE_DIR/fold_0/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=24 \
--learning_rate=5e-5 \
--crf_learning_rate=1e-3 \
--num_train_epochs=10 \
--logging_steps=448 \
--save_steps=448 \
--output_dir=$OUTPUT_DIR/fold_${i}/ \
--overwrite_output_dir \
--overwrite_cache \
--seed=42
done


export BERT_BASE_DIR=./data/user_data/language_model/bert_base
export CLUE_DIR=./data/user_data/5fold_mix
export OUTPUT_DIR=./data/user_data/models/outputs_5fold_mix_span_bert
TASK_NAME='comp'
for i in {0..4}
do
python ./src/pytorch_version/run_ner_span_adv.py \
--model_type=bert \
--model_name_or_path=$BERT_BASE_DIR \
--task_name=$TASK_NAME \
--do_train \
--do_eval \
--do_predict \
--do_adv \
--do_lower_case \
--loss_type=ce \
--data_dir=$CLUE_DIR/fold_0/ \
--train_max_seq_length=512 \
--eval_max_seq_length=512 \
--per_gpu_train_batch_size=8 \
--per_gpu_eval_batch_size=24 \
--learning_rate=5e-5 \
--num_train_epochs=10 \
--logging_steps=448 \
--save_steps=448 \
--warmup_steps=448 \
--output_dir=$OUTPUT_DIR/fold_${i}/ \
--overwrite_output_dir \
--overwrite_cache \
--seed=42
done

CURRENT_DIR=`pwd`
export RUN_PATH=$CURRENT_DIR/src/joleo_code

# nezha large
python $RUN_PATH/main.py \
--maxlen 512  \
--epoch 7	\
--batch_size 3	\
--learning_rate 5e-5	\
--min_learning_rate 1e-5	\
--crf_lr 3666	\
--fold 5	\
--cnt 3	\
--model_name 'nezha'	\
--model_path './data/user_data/model_data/nezha_large_1211'	\
--result_path './data/user_data/models/nezha_large_1211.csv'	\
--config_path './data/user_data/language_model/NEZHA-Large-WWM/bert_config.json'	\
--checkpoint_path './data/user_data/language_model/NEZHA-Large-WWM/model.ckpt-346400'	\
--dict_path './data/user_data/language_model/NEZHA-Large-WWM/vocab.txt'	\
--random_seed 2020

# nezha base
python $RUN_PATH/main.py	\
--maxlen 512	\
--epoch 7	\
--batch_size 6	\
--learning_rate 5e-5	\
--min_learning_rate 1e-5	\
--crf_lr 5000	\
--fold 5	\
--cnt 3	\
--model_name 'nezha'	\
--model_path './data/user_data/model_data/nezha_base_1211'	\
--result_path './data/user_data/models/nezha_base_1211.csv'	\
--config_path './data/user_data/language_model/NEZHA-Base-WWM/bert_config.json'	\
--checkpoint_path './data/user_data/language_model/NEZHA-Base-WWM/model.ckpt-691689'	\
--dict_path './data/user_data/language_model/NEZHA-Base-WWM/vocab.txt'	\
--random_seed 2020

# roberta large
python $RUN_PATH/main.py	\
--maxlen 512	\
--epoch 7	\
--batch_size 3	\
--learning_rate 5e-5	\
--min_learning_rate 1e-5	\
--crf_lr 5000	\
--fold 5	\
--cnt 3	\
--model_name 'bert'	\
--model_path './data/user_data/model_data/roberta_large_1211'	\
--result_path './data/user_data/models/roberta_large_1211.csv'	\
--config_path './data/user_data/language_model/roberta_large/bert_config.json'	\
--checkpoint_path './data/user_data/language_model/roberta_large/bert_model.ckpt'	\
--dict_path './data/user_data/language_model/roberta_large/vocab.txt'	\
--random_seed 2020

# emsemble
python $RUN_PATH/label_ensemble.py \
--result_path './data/user_data/models/blending_crf_result.csv'	\

CURRENT_DIR=`pwd`
export RUN_PATH=$CURRENT_DIR/src/pytorch_version
python $RUN_PATH/postprocess.py
