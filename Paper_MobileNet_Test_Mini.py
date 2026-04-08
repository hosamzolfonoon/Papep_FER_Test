import os
import time
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("Error: tflite-runtime is not installed.")
    print("Install it with: pip install tflite-runtime")
    raise SystemExit

MODEL_PATH = "fer_model.tflite"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --------------------------------------------------
# Check model file
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"Error: model file not found: {MODEL_PATH}")
    raise SystemExit

# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]
output_shape = output_details[0]["shape"]
output_dtype = output_details[0]["dtype"]

print("Model loaded successfully")
print("Input details :", input_details)
print("Output details:", output_details)

if len(input_shape) != 4:
    print("Unexpected input shape.")
    raise SystemExit

model_h = int(input_shape[1])
model_w = int(input_shape[2])
model_c = int(input_shape[3])

# --------------------------------------------------
# Load face detector
# --------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: could not load Haar cascade.")
    raise SystemExit

# --------------------------------------------------
# Open webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open webcam.")
    raise SystemExit

# Optional: lower resolution for better FPS on Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
print("Starting live FER. Press 'q' to quit.")
buffer = []
MAX_SIZE = 60

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read frame.")
        break

    # FPS
    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    buffer.append(fps)
    if len(buffer) > MAX_SIZE:
        buffer.pop(0)
    prev_time = current_time

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        if face_roi.size == 0:
            continue

        # Resize to model input size
        face_resized = cv2.resize(face_roi, (model_w, model_h))

        # Normalize to [0,1]
        face_input = face_resized.astype(np.float32) / 255.0

        # Add channel dimension if needed
        if model_c == 1:
            face_input = np.expand_dims(face_input, axis=-1)

        # Add batch dimension
        face_input = np.expand_dims(face_input, axis=0)

        # Match model input dtype
        if input_dtype == np.float32:
            input_tensor = face_input.astype(np.float32)

        elif input_dtype == np.uint8:
            scale, zero_point = input_details[0]["quantization"]
            if scale and scale > 0:
                input_tensor = face_input / scale + zero_point
                input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)
            else:
                input_tensor = (face_input * 255.0).astype(np.uint8)

        elif input_dtype == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            if scale and scale > 0:
                input_tensor = face_input / scale + zero_point
                input_tensor = np.clip(input_tensor, -128, 127).astype(np.int8)
            else:
                input_tensor = ((face_input * 255.0) - 128).astype(np.int8)

        else:
            print(f"Unsupported input dtype: {input_dtype}")
            continue

        # Run inference
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)[0]

        # Handle quantized output
        if output_dtype == np.uint8:
            scale, zero_point = output_details[0]["quantization"]
            if scale and scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                output_data = output_data.astype(np.float32)

        elif output_dtype == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            if scale and scale > 0:
                output_data = (output_data.astype(np.float32) - zero_point) * scale
            else:
                output_data = output_data.astype(np.float32)

        else:
            output_data = output_data.astype(np.float32)

        pred_index = int(np.argmax(output_data))
        confidence = float(output_data[pred_index])

        if 0 <= pred_index < len(EMOTIONS):
            label = EMOTIONS[pred_index]
        else:
            label = f"Class {pred_index}"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("FER on Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if len(buffer) > 0:
            avg = sum(buffer) / len(buffer)
            print("Average FPS:", avg)
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()