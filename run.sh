#/usr/bin/python
CUDA_VISIBLE_DEVICES=1 python -u  nocontext.py \
	--train_path ../train_docs.pickle \
	--valid_path ../validate_docs.pickle \
	--test_path ../test_docs.pickle \
	--eval_interval 100 \
        --model_path ./mrr_qa_pretrain.md \
	--learning_rate	0.001 \
 	--num_epochs 10 \
	--pretrain_path ../glove.6B.200d.txt \
	--embed_size 200 \
	--cuda \
	--hidden_size 100  2>&1 | tee mrr_qa_pretrain.log
