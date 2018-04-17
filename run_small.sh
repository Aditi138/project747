#/usr/bin/python
CUDA_VISIBLE_DEVICES=1 python -m pudb nocontext.py \
	--train_path ../pickle/small_train_docs.pickle \
	--valid_path ../pickle/small_summaries.pickle \
	--test_path ../pickle/small_summaries.pickle \
	--eval_interval 100 \
        --model_path ./mrr_qa.md \
	--learning_rate	0.001 \
 	--num_epochs 10 \
	--embed_size 512 \
	--cuda \
	--hidden_size 512  2>&1 | tee mrr_qa.log
