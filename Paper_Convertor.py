import tensorflow as tf

MODEL_H5 = "fer2013_big_XCEPTION.54-0.66.hdf5"
MODEL_TFLITE = "fer_model_big.tflite"

model = tf.keras.models.load_model(MODEL_H5, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(MODEL_TFLITE, "wb") as f:
    f.write(tflite_model)

print(f"Saved: {MODEL_TFLITE}")