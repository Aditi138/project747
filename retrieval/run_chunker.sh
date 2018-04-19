#!/usr/bin/env bash
python -u retrieve_chunks.py \
                --input_folder \
                ../../narrativeqa/tmp/ \
                --doc_file \
                ../../narrativeqa/documents.csv \
                --qap_file \
                ../../narrativeqa/qaps.csv \
                --summary_file \
                ../../narrativeqa/summaries.csv \
                --output_folder \
                ../../narrativeqa/chunks/ \
                --ir_model \
                tfidf \
                --num_chunks \
                30 \
                --chunk_size \
                200 \
                --mode test 2>&1 | tee chunking.log
