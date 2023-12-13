### Run end-to-end BEVERS
# setup dataset variable
set -eou pipefail
export PYTHONPATH=.
export DATASET=fever


# activate conda environment
eval "$(conda shell.bash hook)"
conda activate bevers

# prep directories for FEVER
python src/paths.py

# get fever.db and setup conversion
wget -O fever/data/processed/init.db https://fever.ai/download/fever/fever.db
python src/data/convert_db.py

# get jsonl files and prep datasets
wget -O fever/data/raw/train.jsonl https://fever.ai/download/fever/train.jsonl
wget -O fever/data/raw/shared_task_dev.jsonl https://fever.ai/download/fever/shared_task_dev.jsonl
wget -O fever/data/raw/shared_task_test.jsonl https://fever.ai/download/fever/shared_task_test.jsonl
python src/data/parse_dataset.py

# built tf-idf and compute distance scores for each dataset
python src/data/tfidf.py
python src/data/compute_tfidf_distance.py

# measure tf-idf and expand with fuzzy string search
# running fuzzy string search on all data will take ~ 24 hours or more.
python src/eval/measure_tfidf.py

# generate training dataset for sentence selection
python src/data/make_sentence_selection_dataset.py

# train sentence selection model
python src/models/sentence_selection/train_ss_model.py

# run sentence selection
python src/eval/score_sentences_roberta.py \
    --model_path fever/models/sentence_selection/ss_roberta-large_binary_False_epoch=00-valid_accuracy=0.96178.ckpt \
    --model_type roberta-large \
    --save_results


# expand with hyperlinkers (this is currently super inefficient, TODO use something more efficient)
 python src/data/expand_candidates.py 

# run evidence based re-retrieval
python src/eval/score_sentences_roberta.py \
    --model_path fever/models/sentence_selection/ss_roberta-large_binary_False_epoch=00-valid_accuracy=0.96178.ckpt \
    --model_type roberta-large \
    --reretrieval \
    --save_results


# combine original scored sentences with re-retrieval using discounting 
# running this script without --discount factor and with "--dataset dev" will scan over discount factor values
python src/data/combine_sentence_scores.py --discount_factor 0.95


### RoBERTa based classifier 

# # train claim classification model -- RoBERTa Large MNLI
# python src/models/claim_classification/train_cc_model.py

# dump claim files to disk -- RoBERT Large MNLI
# python src/eval/measure_cc_model.py \
#     --ckpt_path fever/models/claim_classification/cc_roberta-large-mnli* \
#     --combined_sentence_scores

### / RoBERTa based classifier

### DeBERTa based classifier

# train claim classification model -- DeBERTa v2 XL MNLI
python src/models/claim_classification/train_cc_model.py \
    --model_type microsoft/deberta-v2-xlarge-mnli \
    --batch_size 2 \
    --gradient_accumulations 8

python src/eval/measure_cc_model.py \
    --model_type microsoft/deberta-v2-xlarge-mnli \
    --ckpt_path /home/mitchell/repos/bevers/fever/models/claim_classification/cc_microsoft-deberta-v2-xlarge-mnli_concat_epoch=01-valid_accuracy=0.81273.ckpt \
    --combined_sentence_scores

### / DeBERTa based classifier

# train xgboost classifier on claim classification outputs
python src/models/xgbc.py

# prepare submission
python src/eval/prep_submission.py
