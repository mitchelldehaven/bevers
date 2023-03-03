# BEVERS - A Simple Pipeline for Fact Verification
<p align="center">
  <img src='bevers.jpeg' width='400'>
</p>
<p align="center"> Image courtesy of Stable Diffusion 2.1 </p>
<hr>

This repo is the code for Baseline fact Extraction and VERification System (BEVERS). The pipeline utilizes standard approaches for each component in the pipeline. Despite it's simplicity, BEVERS achieves SOTA performance on FEVER ([old leaderboard](https://competitions.codalab.org/competitions/18814#results), [new leaderboard](https://codalab.lisn.upsaclay.fr/competitions/7308#results)) and achieves the highest label accuracy F1 score on SciFact ([leaderboard](https://leaderboard.allenai.org/scifact/submissions/public)). 

## Requirements
- conda

## Installation
To create the `bevers` conda environment, Python requirements, and other requirements run the `setup.sh` script. The script requires `sudo` access for setting up SQLite as a fuzzy string search engine.

```
bash setup.sh
```

## Running
There is a run script for FEVER, SciFact, and PubMed. The general pipeline of BEVERS is as follows (PubMed is an exeception):
- TF-IDF setup (and fuzzy string search for FEVER)
- Sentence selection dataset generation, model training, and final dumping of sentence scores.
- Claim classification training and dumping of claim scores.
- Training of XGBoost classifier
- Generating final output files for submission to leaderboards for scoring.

```
# Run FEVER 
bash run_fever.sh
# Run PubMed (NOTE: manual effort is needed here to download required dataset files)
bash run_pubmed.sh
# Run SciFact (running PubMed is a prerequisite here)
bash run_scifact.sh
```

## FEVER Results
| System | Test Label Accuracy| Test FEVER Score |
| ------ | :---------------:  | :-------------:  |
|[LisT5](https://aclanthology.org/2021.acl-short.51/)|79.35|75.87|
|[Stammbach](https://aclanthology.org/2021.fever-1.2/)|79.16|76.78|
|[ProoFVer](https://aclanthology.org/2022.tacl-1.59/)|79.47|76.82|
|Ours (RoBERTa Large MNLI) mixed |79.39 | 76.89|
|Ours (DeBERTa v2 XL MNLI) mixed |**80.24** | **77.70**|

## SciFact Results
| System | SS + L F1 | Abstract Label Only F1 |
| ------ | :------:  | :-------------------:  |
|[VerT5erini](https://aclanthology.org/2021.louhi-1.11/)|58.8|64.9|
|[ASRJoint](https://aclanthology.org/2021.emnlp-main.290/)|63.1|68.1|
|[MultiVers](https://aclanthology.org/2022.findings-naacl.6/)|**67.2** |72.5|
|Ours | 58.1  | **73.2**|

## Todos
- [ ] Release models (still figuring out best way for this, may just use HuggingFace model hub)
- [ ] Finish cleaning up code (started on this but didn't finish)
- [x] Update demo - Done (03/02/23)
- [ ] Some of the code was simply copied from the evaluation repos for ease of use. Properly document source of code that is not mine.

## Potential Todos
- [ ] Improve retrieval for SciFact utilizing neural re-rankers like most other systems do.
- [ ] Release easy to use predictions for sentence selection. This helps people who only want to focus on the claim classification portion of task.

## Regression Tests
In my initial code clean up I changed a decent amount of code and prior to release I wanted to make sure the results were replicable, so I ran regression tests for FEVER and SciFact.
### FEVER
| Run | Test Label Accuracy| Test FEVER Score |
| --- | :----------------: | :--------------: |
| Published (RoBERTa Large MNLI) | 79.39 | 76.89 |
| Regression (02/20/23) | 79.31 | 76.91 |
| Published (DeBERTa v2 XL MNLI) | 80.24 | 77.70 | 
| Regression (02/22/23) | 80.35 | 77.86 | 

### SciFact
| Run | SS + L F1 | Abstract Label Only F1 |
| --- | :-------: | :--------------------: |
| [Published](https://leaderboard.allenai.org/scifact/submission/ccpr8fq1igkl24rohk20)| 58.1  | 73.2|
| [Regression](https://leaderboard.allenai.org/scifact/submission/cfttsdq3t1q51grt1e90) (02/26/23) | 58.3 | 73.8|


## Demo
There is a simple UI for demoing the model. The current setup is a lighter version of what was used in for the best results for reducing compute requriement. 
### Backend
For running the backend Flask API:
```
export DATASET=fever
export PYTHONPATH=.
python demo/src/app.py
```

### Frontend
```
cd demo/frontend
npm i
ng serve
```
There is a simple gif showing the demo below in order to avoid having to setup the demo to see what it does.

<p align="center">
  <img src='fever_demo.gif' width='800' />
</p>
