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

