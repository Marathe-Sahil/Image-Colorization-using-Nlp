from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the colorization model once
def load_model():
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe(
        'model/colorization_deploy_v2.prototxt',
        'model/colorization_release_v2.caffemodel'
    )
    pts = np.load('model/pts_in_hull.npy')
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

net = load_model()

# Colorize function
def colorize_image(image_path, net):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read image")
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (colorized * 255).astype("uint8")

# Sketch effect function
def apply_sketch_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], "input.jpg")
            file.save(path)
            colorized = colorize_image(path, net)
            output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")
            cv2.imwrite(output_path, colorized)
            return render_template("index.html", filename="output.jpg")
    return render_template("index.html")

@app.route("/sketch", methods=["POST"])
def sketch():
    if "cropped_image" not in request.files:
        return "No file uploaded", 400
    file = request.files["cropped_image"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], "crop.jpg")
    file.save(path)
    image = cv2.imread(path)
    if image is None:
        return "Failed to read cropped image", 400
    sketch = apply_sketch_effect(image)
    sketch_path = os.path.join(app.config["UPLOAD_FOLDER"], "sketch.jpg")
    cv2.imwrite(sketch_path, sketch)
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
