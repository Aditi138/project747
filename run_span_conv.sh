#/usr/bin/python
CUDA_VISIBLE_DEVICES=2 python -u span_prediction.py \
	--train_path ../squad/train-v1.1.pickle \
	--valid_path ../squad/dev-v1.1.pickle \
	--test_path ../squad/dev-v1.1.pickle \
	--eval_interval 100 \
        --model_path ./span_squad_conv.md \
	--learning_rate	0.001 \
 	--num_epochs 20 \
	--embed_size 200 \
	--cuda \
	--pretrain_path ../glove.6B.200d.txt \
	--batch_length 20 \
	--job_size 10 \
	--num_layers 2 \
	--hidden_size 100  2>&1 | tee span_squad_conv.log
