from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import urllib.parse, math, pickle
from collections import Counter

# ----- (A)  मॉडल व वेक्टराइज़र लोड करें -----
MODEL_PATH      = "models/phishing_model.h5"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ----- (B)  URL से फीचर निकालने के हेल्पर -----
def calculate_entropy(text):
    counts = Counter(text)
    probs  = [c/len(text) for c in counts.values()]
    return -sum(p * math.log(p) for p in probs)

def extract_features_from_url(url: str):
    parsed   = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    return [
        len(url),                             # length
        url.count("."),                       # num_dots
        url.count("-"),                       # num_hyphens
        url.count("/"),                       # num_slash
        int(parsed.scheme == "https"),        # has_https
        int(parsed.port is not None),         # has_port
        int(len(parsed.query) > 0),           # has_query
        len(parsed.path),                     # path_length
        sum(c.isdigit() for c in url),        # digit_count
        sum(c in "!@#$%^&*()" for c in url),  # special_chars
        int(any(x in hostname for x in ["bit.ly", "goo.gl"])),  # is_shortened
        calculate_entropy(url)                # url_entropy
    ]

def predict_url(url: str):
    text_vec  = vectorizer.transform([url]).toarray().reshape(1, 1, -1)
    features  = np.array([extract_features_from_url(url)])
    score     = model.predict([text_vec, features])[0][0]

    if   score > 0.7: level = "उच्च जोखिम"
    elif score > 0.4: level = "मध्यम जोखिम"
    else:             level = "कम जोखिम"

    return round(float(score), 4), level

# ----- (C)  Flask Routes -----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            score, level = predict_url(url)
            return render_template("result.html",
                                   url=url, score=score, level=level)
    return render_template("index.html")

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
