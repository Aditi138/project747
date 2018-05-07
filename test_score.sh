python -m pudb chunk_score.py \
--train_path ../pickle/smalltrain \
--valid_path ../pickle/smallvalid \
--embed_size 100 \
--hidden_size 128 \
--max_documents 20 \
--max_valid 20 \
--eval_interval 5 \
#--pretrain_path ../embedding/glove100.txt \
