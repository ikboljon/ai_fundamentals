# app.py
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("model", "classifier.joblib")

app = Flask(__name__)

# load model at startup
model_bundle = joblib.load(MODEL_PATH)
vect = model_bundle['vect']
clf = model_bundle['clf']
labels = model_bundle['labels']

def explain_prediction(text, top_k=8):
    """
    Return predicted class, probabilities, and top contributing words for that class.
    Approach: compute TF-IDF vector for text, inspect classifier.coef_ for predicted class,
    multiply TF-IDF values by coef to rank tokens contributing most positively.
    """
    X = vect.transform([text])  # sparse
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    # coefficients for the class
    coef = clf.coef_[pred_idx]  # shape (n_features,)
    # get feature names
    try:
        feature_names = np.array(vect.get_feature_names_out())
    except:
        feature_names = np.array(vect.get_feature_names())
    # compute contribution = tfidf * coef
    x_arr = X.toarray()[0]  # dense for one sample
    contrib = x_arr * coef
    # get top positive contributing tokens
    top_pos_idx = np.argsort(contrib)[-top_k:][::-1]
    top_pos = [(feature_names[i], float(contrib[i]), float(x_arr[i]), float(coef[i])) for i in top_pos_idx if x_arr[i] > 0]
    # also show top negative tokens (words that push away from this class)
    top_neg_idx = np.argsort(contrib)[:top_k]
    top_neg = [(feature_names[i], float(contrib[i]), float(x_arr[i]), float(coef[i])) for i in top_neg_idx if x_arr[i] > 0]
    return {
        'pred_label': pred_label,
        'pred_idx': pred_idx,
        'probs': probs,
        'top_pos': top_pos,
        'top_neg': top_neg
    }

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text","").strip()
        if not text:
            return redirect(url_for('index'))
        return redirect(url_for('result', q=text[:2000]))  # pass preview in URL (limited)
    return render_template("index.html")

@app.route("/result")
def result():
    q = request.args.get("q","")
    text = q
    info = explain_prediction(text, top_k=8)
    # prepare probs display: top 5 classes
    probs = sorted([(labels[i], float(info['probs'][i])) for i in range(len(labels))], key=lambda x: x[1], reverse=True)[:5]
    return render_template("result.html", text=text, pred=info['pred_label'], probs=probs, top_pos=info['top_pos'], top_neg=info['top_neg'])

if __name__ == "__main__":
    app.run(debug=True)
