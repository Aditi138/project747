#/usr/bin/python
CUDA_VISIBLE_DEVICES=1 python -u  nocontext.py \
	--train_path ../lor_codes/narrativeqa/out/small/small_train_docs.pickle \
	--valid_path ../lor_codes/narrativeqa/out/small/small_valid_docs.pickle \
	--test_path ../lor_codes/narrativeqa/out/small/small_test_docs.pickle \
	--eval_interval 100 \
        --model_path ./mrr_qa.md \
	--learning_rate	0.001 \
 	--num_epochs 10 \
	--embed_size 512 \
	--cuda \
	--hidden_size 512  2>&1 | tee mrr_qa.log
