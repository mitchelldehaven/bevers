
### Run end-to-end BEVERS for PubMED finetuning
# setup dataset variable
set -eou pipefail
export PYTHONPATH=.
export DATASET=fever


# activate conda environment
eval "$(conda shell.bash hook)"
conda activate bevers

# download RoBERTa-large-PM-M3 and setup
mkdir models
wget -P models https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz 
tar -vxf models/RoBERTa-large-PM-M3-Voc-hf.tar.gz -C models/
ln -s models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf .

# initial finetune on FEVER
python src/models/sentence_selection/train_ss_model.py --model_type RoBERTa-large-PM-M3-Voc-hf
python src/models/claim_classification/train_cc_model.py --model_type RoBERTa-large-PM-M3-Voc-hf


# prep directories for PubMed
export DATASET=pubmed
python src/paths.py

### TODO: You must manually install PubMedQA-A (`ori_pmqaa.json`) and place to the files into pubmed/data/raw
### as it is on Google drive making wget annoying to use.

# create pubmed nli dataset from pubmed qa
pubmed_datasets="train valid dev"
python src/pubmed/data/prep_pubmed.py
python src/data/tfidf.py \
    --claims_file claims_no_evidence.pkl \
    --tfidf_type cat \
    --datasets $pubmed_datasets

python src/data/compute_tfidf_distance.py \
    --tfidf_type cat \
    --datasets $pubmed_datasets \
    --k 10

# use FEVER fintuned model to give rough first guess at evidence
python src/eval/score_sentences_roberta.py \
    --claims_file claims_no_evidence.pkl \
    --model_path fever/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf* \
    --model_type RoBERTa-large-PM-M3-Voc-hf \
    --datasets $pubmed_datasets \
    --num_labels 3 \
    --save_results \
    --exclude_fuzzy

# # you could potentially run this through again with a better 
# # estimate at evidence, but I found that it doesn't give much improvement

# python src/pubmed/generate_sst_evidence.py --threshold 0.5
# python src/data/make_sentence_selection_dataset.py --exclude_fuzzy

# # train subsequent model on rough pubmed
# python src/models/sentence_selection/train_ss_model.py \
#     --model_type RoBERTa-large-PM-M3-Voc-hf \
#     --checkpoint fever/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf*

# give better guess at evidence on pubmed after finetuning
# python src/eval/score_sentences_roberta.py \
#     --claims_file claims_no_evidence.pkl \
#     --model_path pubmed/models/sentence_selection/ss_RoBERTa-large-PM-M3-Voc-hf* \
#     --model_type RoBERTa-large-PM-M3-Voc-hf \
#     --datasets $pubmed_datasets \
#     --num_labels 3 \
#     --save_results \
#     --exclude_fuzzy
# python src/pubmed/generate_sst_evidence.py --threshold 0.8
