from flask import Flask, jsonify, redirect, render_template, request, url_for
import os
import tempfile

from werkzeug.utils import secure_filename

import label_image


app = Flask(__name__)


STAGE_DESCRIPTIONS = {
    "Verymild Demented": (
        "The Very Mild Demented stage is an early phase of Alzheimer's disease "
        "and is often considered part of Mild Cognitive Impairment. Symptoms "
        "can be subtle but may slightly affect everyday activities."
    ),
    "Mild Demented": (
        "Mild Demented is an early dementia stage where cognitive impairments "
        "become noticeable and may begin to affect daily life, language, "
        "orientation, mood, and behavior."
    ),
    "Moderate Demented": (
        "Moderate Demented refers to a middle stage of cognitive decline with "
        "more significant memory loss, daily activity challenges, and increased "
        "need for support."
    ),
    "Non Demented": (
        "Non Demented indicates no significant cognitive decline or memory "
        "impairment severe enough to interfere with daily life."
    ),
}


@app.route("/")
@app.route("/first")
def first():
    return render_template("index1.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/login")
def login():
    return redirect(url_for("index"))


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    filename = secure_filename(uploaded_file.filename)
    extension = os.path.splitext(filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        file_path = temp_file.name
        uploaded_file.save(file_path)

    try:
        prediction = label_image.predict_with_gradcam(file_path)
        label = prediction["label"].title()
        return jsonify({
            "label": label,
            "description": STAGE_DESCRIPTIONS.get(label, ""),
            "confidence": prediction["confidence"],
            "gradcam_layer": prediction["gradcam_layer"],
            "gradcam_image": prediction["gradcam_image"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
