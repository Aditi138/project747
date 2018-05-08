# project747: NarrativeQA 
## Introduction

This is a PyTorch implementation of our RC model for the NarrativeQA Challenge as presented in "The NarrativeQA Reading Comprehension Challenge" available at https://arxiv.org/pdf/1712.07040.pdf.

We implement several variants of BiDAF (https://arxiv.org/abs/1611.01603) and Tri-Attention model (https://arxiv.org/abs/1803.00191) :-
1. Tri-Attention on Full Summary with ELMo
2. Tri-Attention on Reduced Summary with ELMo
3. No Context with ELMo

We also have the Information Retrieval heuristics in the ../retrieval folder. The ../models folder contains the various models we implemented. The run files for each model are separate are present in the parent folder.
## Quick Start
To run the model:
Download the data using download_stories.sh. Run with the below arguments to pre-process, tokenize and store data locally. The file summaries.csv, qaps.csv, documents.csv, download_stories.sh can be found [here]{https://github.com/deepmind/narrativeqa}.

To train the tri-attention model, run with the following parameters:-
```
 git checkout tri_attn
 MODEL_NAME="triattn_elmo"
 python -u  tri_attn.py 
          --train_path ../train_summaries_embed.pickle \
          --valid_path ../valid_summaries_embed.pickle \ 
          --test_path ../test_summaries_embed.pickle \  
          --eval_interval 500 \ 
          --model_path ./${MODEL_NAME}.md \
          --learning_rate 0.001 \ 
          --num_epochs 10 \
          --cuda \    
          --elmo \ 
          --embed_size 1024 \  
          --batch_length 20 \    
          --job_size 10 \   
          --dropout 0.4 \
          --s_file ${MODEL_NAME}.s_file.test.txt \   
          --debug_file ${MODEL_NAME}.debug  \
          --hidden_size 128  2>&1 | tee ${MODEL_NAME}.test.log
```
To run with the reduced summary specify argument ``` --reduced ``` and remove the argument ```--elmo ```.
We pre-compute the ElMo embeddings for our dataset, to get the input run - 
```
python -u precompute_elmo.py 
    --pickle_folder ../<output_folder> \
    --qaps_file  ../qaps.csv \
    --summary_file ../summary.csc \
    --doc_file ../documents.csv \
    --use_doc
```
NOTE: For precomputing ElMo, please follow the instructions to set up the environment here (https://github.com/allenai/allennlp)
