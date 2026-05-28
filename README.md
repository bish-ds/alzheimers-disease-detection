---
title: Alzheimer's Disease Detection
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Alzheimer's Disease Detection

Flask web application for classifying brain scan images into Alzheimer's disease stage categories using a retrained TensorFlow image classification graph. The prediction page also returns a Grad-CAM overlay to show the image regions most associated with the predicted class.

The app supports four labels:

- Mild Demented
- Moderate Demented
- Non Demented
- Very Mild Demented

> Educational project only. This application is not a medical device and should not be used as a substitute for clinical diagnosis.

## Project Structure

```text
.
|-- app.py
|-- label_image.py
|-- retrain.py
|-- retrained_graph.pb
|-- retrained_labels.txt
|-- static/
`-- templates/
```

## How It Works

1. `app.py` runs the Flask web app.
2. The upload page sends an image to `/predict`.
3. `label_image.py` loads `retrained_graph.pb` and `retrained_labels.txt`.
4. TensorFlow classifies the uploaded image and returns the predicted label, confidence score, short description, and Grad-CAM overlay.
5. `retrain.py` can be used to retrain the image classifier with a folder-based dataset.

## Setup

Use Python 3.8, 3.9, or 3.10 for best compatibility with the pinned TensorFlow version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

The prediction page is available directly at `/index`; no login is required.

## Retraining

Prepare the dataset with one folder per class:

```text
dataset/
|-- mild demented/
|-- moderate demented/
|-- non demented/
`-- verymild demented/
```

Run:

```bash
python retrain.py --image_dir dataset --output_graph retrained_graph.pb --output_labels retrained_labels.txt
```

## Publishing To GitHub

The trained model file is large:

```text
retrained_graph.pb - about 87 MB
```

For a cleaner GitHub repository, the model should be stored with GitHub Releases, Git LFS, or the deployed Hugging Face Space instead of being committed directly. The app looks for `retrained_graph.pb` in the project root first, then falls back to `models/retrained_graph.pb`.

Basic Git commands:

```bash
git init
git add .
git commit -m "Initial Alzheimer's disease detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Notes Before Public Release

- Remove generated folders such as `__pycache__/` and `.ipynb_checkpoints/` from commits.
- Replace placeholder phone numbers, addresses, emails, and author names in the templates.
- Check template spelling from `Alzhimer` to `Alzheimer`.
- Remove unrelated stock images or template assets that are not part of the project.
- Keep the medical disclaimer visible if the app is shared publicly.
- Add dataset source, training details, metrics, and limitations so reviewers understand model performance.

## License

No license has been specified yet. Add a license before publishing if others should be allowed to use or modify the code.
