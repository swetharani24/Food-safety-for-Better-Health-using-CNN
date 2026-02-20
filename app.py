import os
import json
import numpy as np
import redis
import cv2
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from PIL import Image

# ------------------ Constants ------------------
FOOD_CLASSES = [
    "dal_makhani","idli","masala_dosa","omelette","donut","chole_bhature",
    "ice_cream","crispy_chicken","pav_bhaji","hot_dog","kaathi_rolls",
    "cheesecake","baked_potato","chapati","pakode","fried_rice","dhokla",
    "sandwich","chicken_curry","jalebi","samosa","apple_pie","paani_puri",
    "taquito","kulfi","pizza","burger","kadai_paneer","butter_naan","sushi",
    "fries","taco","chai","momos"
]

MODEL_PATHS = {
    "CNN": "models/cnn_model_34classes.h5",
    "VGG16": "models/vgg16model.h5",
    "ResNet50": "models/resnet50_food_34classes.h5"
}

MODEL_INPUT_SIZES = {
    "CNN": (256, 256),
    "VGG16": (224, 224),
    "ResNet50": (256, 256)
}

# ------------------ Flask & Redis ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# ------------------ Load class indices ------------------

# ------------------ Class indices ------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# index -> class
CLASS_NAMES = {v: k for k, v in class_indices.items()}

# class list for UI (RIGHT PANEL) – SAME ORDER AS TRAINING
FOOD_CLASSES = [None] * len(class_indices)
for name, idx in class_indices.items():
    FOOD_CLASSES[idx] = name


# ------------------ Load Models ------------------
models = {}
for name, path in MODEL_PATHS.items():
    try:
        models[name] = load_model(path, compile=False)
        print(f"{name} loaded successfully")
    except Exception as e:
        print(f"{name} not loaded:", e)

# ------------------ Preprocessing per model ------------------
def preprocess_cnn(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(MODEL_INPUT_SIZES["CNN"])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def preprocess_vgg(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(MODEL_INPUT_SIZES["VGG16"])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = vgg_preprocess(x)
    return x

def preprocess_resnet(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(MODEL_INPUT_SIZES["ResNet50"])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet_preprocess(x)
    return x

# ------------------ Prediction per model ------------------
def predict_cnn(model, img):
    x = preprocess_cnn(img)
    preds = model.predict(x)
    pred_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES.get(pred_index, "Unknown")

    confidence = round(float(np.max(preds))*100, 2)
    return predicted_class, confidence

def predict_vgg(model, image):

    # image is already a PIL Image
    img = np.array(image)

    # make sure it is RGB
    if img.shape[-1] == 4:   # RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (224, 224))

    img = img.astype("float32") / 255.0   # same as training

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)

    pred_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    predicted_class = CLASS_NAMES.get(pred_index, "Unknown")

    return predicted_class, confidence

def predict_resnet(model, img):
    x = preprocess_resnet(img)
    preds = model.predict(x)
    pred_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES.get(pred_index, "Unknown")
    confidence = round(float(np.max(preds))*100, 2)
    return predicted_class, confidence

# ------------------ Nutrition from Redis ------------------
def get_food_from_redis(food_name):
    if not food_name:
        return {}
    key = f"food:{food_name.lower().replace(' ', '_')}"  # match Redis keys
    data = redis_client.get(key)
    if not data:
        return {"calories":"N/A","carbs":"N/A","protein":"N/A","fat":"N/A","fiber":"N/A","recommended":"Unknown"}
    raw = json.loads(data)
    return {
        "calories": raw.get("calories_kcal","N/A"),
        "carbs": raw.get("total_carbohydrates_g","N/A"),
        "protein": raw.get("protein_g","N/A"),
        "fat": raw.get("fat_g","N/A"),
        "fiber": raw.get("fiber_g","N/A"),
        "recommended": "Yes" if raw.get("category")=="veg" else "No"
    }

# ------------------ Metrics from Redis ------------------
def load_cnn_metrics():
    data = redis_client.get("model:cnn:metrics")
    if not data:
        return None

    metrics = json.loads(data)

    return {
        "train_accuracy": metrics.get("training_accuracy",0),
        "val_accuracy": metrics.get("validation_accuracy",0),
        "test_accuracy": metrics.get("test_accuracy",0),
        "classification_report": metrics.get("classification_report", ""),
        "precision": "N/A",
        "recall": "N/A",
        "f1": "N/A"
    }


def load_vgg_metrics():
    data = redis_client.get("model:vgg16:metrics")
    if not data:
        return None

    metrics = json.loads(data)

    precision = recall = f1 = "N/A"

    cr = metrics.get("classification_report")
    if isinstance(cr, dict) and "weighted avg" in cr:
        precision = cr["weighted avg"].get("precision", "N/A")
        recall    = cr["weighted avg"].get("recall", "N/A")
        f1        = cr["weighted avg"].get("f1-score", "N/A")

    return {
        "train_accuracy": metrics.get("manual_train_accuracy", 0),
        "val_accuracy":   metrics.get("manual_validation_accuracy", 0),
        "test_accuracy":  metrics.get("manual_test_accuracy", 0),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def load_resnet_metrics():
    data = redis_client.get("model:resnet50:metrics")
    if not data:
        return None

    metrics = json.loads(data)

    precision = recall = f1 = "N/A"

    cr = metrics.get("classification_report")
    if isinstance(cr, dict) and "weighted avg" in cr:
        precision = cr["weighted avg"].get("precision", "N/A")
        recall    = cr["weighted avg"].get("recall", "N/A")
        f1        = cr["weighted avg"].get("f1-score", "N/A")

    return {
        "train_accuracy": metrics.get("train_accuracy", 0),
        "val_accuracy":   metrics.get("val_accuracy", 0),
        "test_accuracy":  metrics.get("test_accuracy", 0),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ------------------ Flask Route ------------------
@app.route("/", methods=["GET","POST"])
def index():
    cnn_metrics = load_cnn_metrics()
    vgg_metrics = load_vgg_metrics()
    resnet_metrics = load_resnet_metrics()

    show_result = False
    predicted_class = None
    confidence = None
    nutrition = {}
    recommended = None
    actual_class = None
    filename = None
    model_selected = None
    metrics = None

    if request.method=="POST":
        file = request.files.get("image")
        actual_class = request.form.get("actual_class","")
        model_selected = request.form.get("model")

        if file and model_selected in models:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            uploaded_image = Image.open(save_path)

            # --- Predict with correct model & preprocessing ---
            if model_selected=="CNN":
                predicted_class, confidence = predict_cnn(models["CNN"], uploaded_image)
                metrics = cnn_metrics
            elif model_selected=="VGG16":
                predicted_class, confidence = predict_vgg(models["VGG16"], uploaded_image)
                metrics = vgg_metrics
            elif model_selected=="ResNet50":
                predicted_class, confidence = predict_resnet(models["ResNet50"], uploaded_image)
                metrics = resnet_metrics

            nutrition = get_food_from_redis(predicted_class)
            recommended = nutrition.get("recommended","Unknown")
            show_result = True

    return render_template(
        "index.html",
        cnn_metrics=cnn_metrics,
        vgg_metrics=vgg_metrics,
        resnet_metrics=resnet_metrics,
        food_classes=FOOD_CLASSES,
        available_models=list(models.keys()),
        model_selected=model_selected,
        show_result=show_result,
        predicted_class=predicted_class,
        confidence=confidence,
        nutrition=nutrition,
        recommended=recommended,
        actual_class=actual_class,
        filename=filename,
        image_url=url_for('static', filename='uploads/' + filename) if filename else None,
        train_accuracy=metrics["train_accuracy"] if metrics else "N/A",
        val_accuracy=metrics["val_accuracy"] if metrics else "N/A",
        test_accuracy=metrics["test_accuracy"] if metrics else "N/A",
        precision = metrics["precision"] if metrics and "precision" in metrics else "N/A",
        recall = metrics["recall"] if metrics and "recall" in metrics else "N/A",
        f1_score=metrics["f1"] if metrics and "f1" in metrics else "N/A",
        classification_report=metrics.get("classification_report", "") if metrics else ""

    )

# ------------------ Run App ------------------
if __name__=="__main__":
    app.run(debug=True)
