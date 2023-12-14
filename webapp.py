import argparse
import io
import os
from PIL import Image, ImageDraw, ImageFont
import datetime

import torch
from flask import Flask, render_template, request, redirect


app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)

    # Menggunakan font TrueType dengan tebal dan ukuran yang diinginkan
    font = ImageFont.truetype("arial.ttf", size=20, encoding="unic")

    num_objects = 0  # Inisialisasi jumlah objek terdeteksi

    for pred in predictions.xyxy[0]:
        if pred[4] >= threshold:
            num_objects += 1  # Hitung objek yang terdeteksi
            box = [int(coord) for coord in pred[:4]]
            label = f"{model.names[int(pred[5])]} {pred[4]:.2f}"

            # # Menampilkan teks dengan font tebal dan ukuran yang diinginkan
            # draw.rectangle(box, outline="red", width=2)
            # draw.text((box[0], box[1]), label, fill="green", font=font)

    # Menampilkan jumlah objek yang terdeteksi pada gambar
    draw.text((10, 10), f"Jumlah Objek: {num_objects}", fill="green", font=font)

    return image, num_objects

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("error.html", error_message="Masukkan Gambar yang ingin di deteksi!!!")

        file = request.files["file"]
        if file.filename == '':
            return render_template("error.html", error_message="Masukkan Gambar yang ingin di deteksi!!!")

        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            return render_template("error.html", error_message=str(e))

        try:
            results = model([img])
            results.render()
            now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
            img_savename = f"static/{now_time}.png"

            img_with_boxes, num_objects = draw_boxes(Image.fromarray(results.ims[0].copy()), results, threshold=0.5)
            img_with_boxes.save(img_savename)

            return render_template("hasil.html", result_image=img_savename, num_objects=num_objects)

        except Exception as e:
            return render_template("error.html", error_message=str(e))

    return render_template("dashboard.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(debug=True)
    # app.run(debug=False, host='0.0.0.0')
    # app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
