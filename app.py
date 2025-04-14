from flask import Flask, request, render_template
from transformers import pipeline
import torch

app = Flask(__name__)

classifier = pipeline(
    "text-classification",
    model="finiteautomata/bertweet-base-sentiment-analysis",
    device=0 if torch.cuda.is_available() else -1
)

LABEL_MAP = {
    "POS": "POSITIVE 😊",
    "NEG": "NEGATIVE 😠", 
    "NEU": "NEUTRAL 😐"
}

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    if request.method == "POST":
        text = request.form["text"]
        result = classifier(text)[0]
        sentiment = {
            "label": LABEL_MAP.get(result["label"], result["label"]),
            "score": result["score"]
        }
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True, port=5000)