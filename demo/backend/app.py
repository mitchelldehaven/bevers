from http import HTTPStatus

from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from src.models.fever_model import FeverModel

app = Flask(__name__)
CORS(app)

fever_model = FeverModel()

@app.route("/api/get_candidate_sentences", methods=["GET"])
def get_candidate_sentences():
    """Returns candidate sentences for a given claim query."""
    if not request.args:
        abort(HTTPStatus.BAD_REQUEST)
    claim = request.args.get("claim")
    top_docs = fever_model.retrieve_docs(claim, 5)
    sentences, sentence_ids = fever_model.get_sentences(top_docs)
    evidence = [{"title": title, "line": line} for title, line in sentences]
    response = {"evidence_list": evidence, "evidence_ids": sentence_ids}
    return response


@app.route("/api/get_top5_sentences", methods=["POST"])
def get_top5_sentences():
    """Returns top5 scoring sentences given a list of candidate sentences."""
    necessary_keys = ["claim", "sentences"]
    if not request.json or all(key in request.json for key in necessary_keys):
        abort(HTTPStatus.BAD_REQUEST)
    
    claim = request.json["params"]["claim"]
    sentences = request.json["params"]["sentences"]
    sentence_ids = request.json["params"]["sentence_ids"]
    sentences = [(s["title"], s["line"]) for s in sentences]
    top5_sentences, top5_ids, top5_scores = fever_model.get_topk_sentences(claim, sentences, sentence_ids)
    top5_evidence = [{"title": title, "line": line} for title, line in top5_sentences]
    response = {
        "top5_evidence": top5_evidence,
        "top5_ids": top5_ids,
        "top5_scores": top5_scores
    }
    return response


@app.route("/api/get_prediction", methods=["POST"])
def get_prediction():
    """Returns claim prediction given top 5 candidate sentences."""
    necessary_keys = ["claim", "top5_sentences"]
    if not request.json or all(key in request.json for key in necessary_keys):
        abort(HTTPStatus.BAD_REQUEST)
    pred_num_to_string = {
        0: "REFUTES",
        1: "NOT ENOUGH INFO",
        2: "SUPPORTS"
    }
    claim = request.json["params"]["claim"]
    top5_sentences = request.json["params"]["top5_sentences"]
    top5_sentences = [(s["title"], s["line"]) for s in top5_sentences]
    pred_dist = fever_model.predict_claim_with_sentences(claim, top5_sentences).numpy()[0]
    pred = int(pred_dist.argmax())
    pred_string = pred_num_to_string[pred]
    response = {
        "prediction_vector": pred_dist.tolist(),
        "prediction": pred_num_to_string[pred],
        "prediction_num": pred
    }
    return response

if __name__ == "__main__":
    app.run(threaded=False)
