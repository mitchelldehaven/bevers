yes | conda create -n bevers python=3.8
eval "$(conda shell.bash hook)"
conda activate bevers
yes | conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
python -m spacy download en_core_web_trf
# for fuzzy string search
sudo apt-get install libsqlite3-dev
pip install git+https://github.com/karlb/sqlite-spellfix
# for standard idf in tf-idf
pip install git+https://github.com/mitchelldehaven/scikit-learn
# for allowing loss function parameters passed to model
pip install git+https://github.com/mitchelldehaven/transformers
