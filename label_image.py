from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import io
import os

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


MODEL_FILE = os.environ.get("MODEL_PATH", "retrained_graph.pb")
if not os.path.exists(MODEL_FILE) and os.path.exists("models/retrained_graph.pb"):
    MODEL_FILE = "models/retrained_graph.pb"

LABEL_FILE = os.environ.get("LABEL_PATH", "retrained_labels.txt")
INPUT_HEIGHT = 299
INPUT_WIDTH = 299
INPUT_MEAN = 128
INPUT_STD = 128
INPUT_LAYER = "Mul"
OUTPUT_LAYER = "final_result"
GRADCAM_LAYER = os.environ.get("GRADCAM_LAYER")


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    labels = []
    with open(label_file, "r", encoding="utf-8") as label_handle:
        for line in label_handle.readlines():
            labels.append(line.rstrip())
    return labels


def load_image_array(file_name):
    with Image.open(file_name) as image:
        image = image.convert("RGB")
        original = image.copy()
        resized = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)

    image_array = np.asarray(resized).astype(np.float32)
    normalized = (image_array - INPUT_MEAN) / INPUT_STD
    normalized = np.expand_dims(normalized, axis=0)
    return normalized, original


def image_to_data_url(image, image_format="JPEG"):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
    return "data:{};base64,{}".format(mime, encoded)


def colorize_heatmap(heatmap):
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0.0, 1.0)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


def overlay_heatmap(original_image, heatmap, alpha=0.45):
    heatmap_image = Image.fromarray(colorize_heatmap(heatmap)).resize(
        original_image.size, Image.BILINEAR
    )
    return Image.blend(original_image.convert("RGB"), heatmap_image, alpha)


def find_gradcam_tensor(graph):
    if GRADCAM_LAYER:
        return graph.get_tensor_by_name(GRADCAM_LAYER)

    preferred_names = [
        "import/mixed_10/join:0",
        "import/mixed_9/join:0",
        "import/mixed_8/join:0",
    ]
    for tensor_name in preferred_names:
        try:
            return graph.get_tensor_by_name(tensor_name)
        except KeyError:
            pass

    for op in reversed(graph.get_operations()):
        if op.type not in ("ConcatV2", "Concat", "Conv2D", "Relu", "Relu6"):
            continue
        for tensor in op.outputs:
            shape = tensor.shape.as_list()
            if len(shape) == 4:
                return tensor

    raise ValueError("Could not find a 4D convolution tensor for Grad-CAM.")


def build_gradcam(conv_values, gradient_values):
    conv_values = conv_values[0]
    gradient_values = gradient_values[0]
    weights = np.mean(gradient_values, axis=(0, 1))
    heatmap = np.sum(conv_values * weights, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def predict_with_gradcam(image_path):
    graph = load_graph(MODEL_FILE)
    labels = load_labels(LABEL_FILE)
    tensor, original_image = load_image_array(image_path)

    input_tensor = graph.get_tensor_by_name("import/{}:0".format(INPUT_LAYER))
    output_tensor = graph.get_tensor_by_name("import/{}:0".format(OUTPUT_LAYER))
    gradcam_tensor = find_gradcam_tensor(graph)
    with graph.as_default():
        class_index_input = tf.placeholder(tf.int32, shape=())
        class_score = tf.gather(output_tensor[0], class_index_input)
        gradients = tf.gradients(class_score, gradcam_tensor)[0]
        if gradients is None:
            raise ValueError(
                "Could not compute Grad-CAM gradients for {}.".format(
                    gradcam_tensor.name
                )
            )

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(output_tensor, {input_tensor: tensor})
        predictions = np.squeeze(predictions)
        class_index = int(np.argmax(predictions))
        conv_values, gradient_values = sess.run(
            [gradcam_tensor, gradients],
            {input_tensor: tensor, class_index_input: class_index},
        )

    heatmap = build_gradcam(conv_values, gradient_values)
    overlay = overlay_heatmap(original_image, heatmap)

    return {
        "label": labels[class_index],
        "confidence": float(predictions[class_index]),
        "gradcam_layer": gradcam_tensor.name,
        "gradcam_image": image_to_data_url(overlay),
    }


def main(img):
    return predict_with_gradcam(img)["label"]
