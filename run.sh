#/usr/bin/python
MODEL_NAME="triattn_with_scoring_adding_contextlayer"
python -u   tri_attn_sentence_scoring.py \
	--train_path ../pickle/smalltrain_summaries_embed.pickle \
	--valid_path ../pickle/smalltest_summaries_embed.pickle \
	--test_path ../pickle/smalltest_summaries_embed.pickle \
	--eval_interval 20 \
        --model_path ./${MODEL_NAME}.md \
	--learning_rate	0.002 \
	--seed 1234 \
 	--num_epochs 20 \
	--cuda \
	--embed_size 300 \
	--batch_length 1 \
	--job_size 10 \
	--num_layers 2 \
	--dropout 0.2 \
	--dropout_emb 0.2 \
	--max_documents 20 \
	--hidden_size 128  2>&1 | tee ${MODEL_NAME}.log


		# --pretrain_path ../embedding/glove.840B.300d.txt \

