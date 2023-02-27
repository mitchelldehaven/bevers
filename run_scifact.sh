### Run end-to-end BEVERS for SciFact finetuning
# setup dataset variable
set -eou pipefail
export PYTHONPATH=.
export DATASET=scifact

# activate conda environment
eval "$(conda shell.bash hook)"
conda activate bevers

# prepare scifact directory
python src/paths.py

# download scifact data and prepare
wget -P scifact/data/raw https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
tar -xvf scifact/data/raw/data.tar.gz -C scifact/data/raw
mv scifact/data/raw/data/* scifact/data/raw
rm scifact/data/raw/data/ -r
python src/scifact/data/prep_scifact_data.py

# set up tfidf
python src/data/run_tfidf_expts.py
python src/eval/measure_doc_recall.py -- best config is tfidf_type == cat and expt4.json best config
python src/data/tfidf.py \
    --tfidf_type cat \
    --config_file scifact/expts/conf/expt4.json
python src/data/compute_tfidf_distance.py \
    --tfidf_type cat \
    --k 10
python src/data/compute_tfidf_distance.py \
    --tfidf_type cat \
    --k 20

# make sentence selection dataset
python src/data/make_sentence_selection_dataset.py \
    --ks 5 10 20 40 80 120 \
    --exclude_fuzzy


# use pubmed model as base model for finetuning sentence selection model
python src/models/sentence_selection/train_ss_model.py \
    --checkpoint pubmed/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf_binary_False_epoch\=00-valid_accuracy\=0.99068.ckpt \
    --model_type RoBERTa-large-PM-M3-Voc-hf \
    --epochs 10 \
    --valid_interval 1.0 \
    --ss_dataset_tag k_120_random_expanded \
    --batch_size 8 \
    --gradient_accumulations 2 \
    --tokenizer_max_len 512 


python src/eval/score_sentences_roberta.py \
    --model_path scifact/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf_binary_False_epoch=02-valid_accuracy=0.96956.ckpt \
    --model_type RoBERTa-large-PM-M3-Voc-hf \
    --save_results \
    --exclude_fuzzy \
    --k 20

# use FEVER finetuned RoBERTa-large-PM-M3-Voc as starting point
python src/models/claim_classification/train_cc_model.py \
    --checkpoint fever/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf_binary_False_epoch=00-valid_accuracy=0.95353.ckpt \
    --model_type RoBERTa-large-PM-M3-Voc-hf \
    --epochs 10 \
    --sentence_scores_files sentence_scores.npy \
    --valid_interval 1.0 \
    --batch_size 8 \
    --gradient_accumulations 2

# generate claim scores
python src/eval/measure_cc_model.py \
    --model_type RoBERTa-large-PM-M3-Voc-hf \
    --ckpt_path /home/mitchell/repos/bevers/scifact/models/claim_classification/cc_RoBERTa-large-PM-M3-Voc-hf_concat_epoch=01-valid_accuracy=0.83000.ckpt \

# build gradient boosting classifier
python src/models/xgbc.py --claim_file claim_scores.npy

# dump `predictions.jsonl` to scifact/outputs
python src/scifact/eval/prep_submission.py
