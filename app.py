import os
import json
import numpy as np
from flask import Flask, request, render_template_string
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Maks. 8 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "batikin_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

# Muat model dan daftar kelas ketika aplikasi mulai.
model = load_model(MODEL_PATH, compile=False)

with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

if model.output_shape[-1] != len(class_names):
    raise ValueError("Jumlah output model tidak cocok dengan daftar kelas.")

PAGE = """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BatikiN — Identifikasi Motif Batik</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: Arial, sans-serif;
      background: #f7f2e9; color: #30251e;
    }
    header {
      background: #572e22; color: #fff8ed;
      padding: 24px 7%; 
    }
    header h1 { margin: 0 0 8px; }
    header p { margin: 0; }
    main {
      max-width: 760px; margin: 42px auto;
      padding: 0 20px;
    }
    .panel {
      background: white; padding: 28px;
      border-radius: 18px; box-shadow: 0 5px 20px #0000000c;
    }
    input, button {
      width: 100%; margin-top: 14px; padding: 13px;
      border-radius: 9px; font-size: 16px;
    }
    input { border: 1px solid #cfc4b6; }
    button {
      border: 0; color: white; background: #794431;
      font-weight: bold; cursor: pointer;
    }
    button:hover { background: #572e22; }
    .result {
      margin-top: 24px; padding: 18px;
      background: #f7f2e9; border-radius: 12px;
    }
    .error { color: #a12525; }
    .muted { color: #766b61; line-height: 1.6; }
  </style>
</head>
<body>
  <header>
    <h1>BatikiN</h1>
    <p>Jelajahi keindahan motif batik Nusantara</p>
  </header>
  <main>
    <section class="panel">
      <h2>Identifikasi Motif Batik</h2>
      <p class="muted">
        Unggah gambar batik untuk melihat prediksi kelas motif
        dari model CNN BatikiN.
      </p>
      <form method="post" enctype="multipart/form-data">
        <label for="image">Pilih gambar batik (maksimal 8 MB)</label>
        <input id="image" name="image" type="file"
               accept="image/png,image/jpeg,image/webp" required>
        <button type="submit">Identifikasi Motif</button>
      </form>

      {% if error %}
        <div class="result error">{{ error }}</div>
      {% endif %}

      {% if result %}
        <div class="result">
          <h3>Hasil Prediksi</h3>
          <p><strong>Motif:</strong> {{ result.name }}</p>
          <p><strong>Skor model:</strong> {{ result.confidence }}%</p>
          <p class="muted">
            Skor ini merupakan keluaran model, bukan jaminan bahwa
            identifikasi selalu benar.
          </p>
        </div>
      {% endif %}
    </section>
  </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        if uploaded is None or not uploaded.filename:
            error = "Silakan pilih gambar terlebih dahulu."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                image = image.resize((224, 224))

                # Sesuai preprocessing MobileNetV2 saat training.
                array = np.asarray(image, dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                array = preprocess_input(array)

                probabilities = model.predict(array, verbose=0)[0]
                index = int(np.argmax(probabilities))

                result = {
                    "name": class_names[index],
                    "confidence": round(float(probabilities[index]) * 100, 2),
                }

            except (UnidentifiedImageError, OSError):
                error = "File tidak dapat dibaca sebagai gambar. Coba file JPG atau PNG."
            except Exception:
                app.logger.exception("Prediksi gagal")
                error = "Prediksi gagal. Silakan coba lagi nanti."

    return render_template_string(PAGE, result=result, error=error)


@app.errorhandler(413)
def too_large(_error):
    return "Ukuran gambar melebihi batas 8 MB.", 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
