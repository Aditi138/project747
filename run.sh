#/usr/bin/python
CUDA_VISIBLE_DEVICES=1 python -u  main.py \
	--train_path ../lor_codes/narrativeqa/out/out/train_docs.pickle \
	--valid_path ../lor_codes/narrativeqa/out/out/validate_docs.pickle \
	--test_path ../lor_codes/narrativeqa/out/out/test_docs.pickle \
	--eval_interval 100 \
        --model_path ./mrr_qa.md \
	--learning_rate	0.001 \
 	--num_epochs 10 \
	--embed_size 512 \
	--cuda \
	--hidden_size 512  2>&1 | tee mrr_qa.log
