base_folder="/home/michiel/Dropbox/CMU/747/project/narrativeqa-master/"

input_extension="rawdata/"
summary_extension="third_party/wikipedia/summaries.csv"
question_extension="qaps.csv"
document_extension="documents.csv"
pickle_extension="pickle/"

python prepare_data.py \
--input_folder $base_folder$input_extension \
--summary_path $base_folder$summary_extension \
--qap_path $base_folder$question_extension \
--document_path $base_folder$document_extension \
--pickle_folder $base_folder$pickle_extension \
--small_number 10