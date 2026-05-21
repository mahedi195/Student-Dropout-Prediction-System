import os
import tensorflow as tf
import joblib
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, "dropout", "dropout_model.keras")
scaler_path = os.path.join(settings.BASE_DIR, "dropout", "scaler.pkl")
imputer_path = os.path.join(settings.BASE_DIR, "dropout", "imputer.pkl")

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)
imputer = joblib.load(imputer_path)