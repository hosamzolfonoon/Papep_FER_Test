import cv2
import time
import pandas as pd
import numpy as np
import warnings
import pandas as pd
import joblib
import cv2
import mediapipe as mp

warnings.filterwarnings("ignore")

class FaceMechDetector():
    def __init__(self, refine_landmarks=True,
                 staticMode=True, max_number_face=1,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.6):
        self.staticMode = staticMode
        self.max_number_face = max_number_face
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,
                                                 self.max_number_face,
                                                 self.refine_landmarks,
                                                 self.min_tracking_confidence,
                                                 self.min_detection_confidence)

    def findFaceMech (self, img, draw=True):
        face_landmarks_dict = {}

        if draw :
            self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceMesh.process(self.imgRGB)

            if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    height, width, channel = img.shape # it very critical point: it is equal to y, x, z not x, y, z
                    for id, lm in enumerate(faceLms.landmark):
                        #x, y, z = int(lm.x*width), int(lm.y*height), int(lm.z*channel)
                        x, y = int(lm.x*width), int(lm.y*height)
                        #face_landmarks_dict[id] = [x, y, z]
                        face_landmarks_dict[id] = [x, y]
        return face_landmarks_dict

detector_face = FaceMechDetector(staticMode=False)


def normalize_list(coord_list):
    """
    Normalize a list of coordinates by translating and scaling.
    Args:
        coord_list (list): List of coordinates (x or y).
    Returns:
        list: Normalized coordinates in the range (-1, 1).
    """
    # Subtract the reference point (first element in the list)
    translated = [coord - coord_list[1] for coord in coord_list]

    # Scale by the maximum absolute value
    max_value = max(abs(coord) for coord in translated) if translated else 1  # Avoid division by zero
    normalized = [coord / max_value for coord in translated]
    return normalized

#fair
def predict_df_generator_face(img):
    prediction_message = ''
    faceLandmrksDict = detector_face.findFaceMech(img)
    detection_list = [
        0, 5, 10, 13, 14, 17, 32, 39, 49, 50,
        55, 61, 63, 66, 70, 80, 84, 88, 91, 105,
        107, 118, 145, 152, 159, 175, 181, 199, 200, 206,
        208, 212, 216, 234, 262, 269, 279, 280, 285, 291,
        293, 296, 300, 311, 314, 321, 334, 336, 347, 374,
        386, 402, 405, 426, 428, 432, 436, 454, 468, 473
        ]
    column_list_items = []
    for i in detection_list:
        column_list_items.append(str(i) + '_X')
        column_list_items.append(str(i) + '_Y')
    column_list_items.append('emotion')
    empty_array = np.zeros((1, len(column_list_items)-1)) # 'column_list' function includes 'emotion' which should be remove in features extraxtion
    j = 0
    if len(faceLandmrksDict) == 478:
        raw_list_x = []
        raw_list_y = []
        for item in detection_list:
            raw_list_x.append(faceLandmrksDict[item][0])
            raw_list_y.append(faceLandmrksDict[item][1])
        normalized_raw_list_x = normalize_list(raw_list_x)
        normalized_raw_list_y = normalize_list(raw_list_y)
        for i in range(len(detection_list)):
            empty_array[j, i*2] = normalized_raw_list_x[i]
            empty_array[j, i*2+1] = normalized_raw_list_y[i]
        
    else:
        prediction_message = 'No Face'
    X_img_face = pd.DataFrame(empty_array, columns=column_list_items[:-1]) # 'column_list' function includes 'emotion' which should be remove in features extraxtion
    return X_img_face, prediction_message

def prediction_face(model_svm, X_img_face, prediction_message):
    prediction_message_list = []
    if prediction_message != 'No Face':
        emotion_folder_list = ['Angry', 'Fear', 'Happy', 'Normal', 'Sad', 'Surprise']
        pred_svm = int(model_svm.predict(X_img_face))
        prediction_message_list.append('SVM : '+str(emotion_folder_list[pred_svm]))
    else:
        prediction_message_list.append(prediction_message)
    return prediction_message_list

def cap_setting():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def fps_message(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

def put_text_face(img, predict_message_final):
    for i in range(len(predict_message_final)):
        cv2.putText(img, predict_message_final[i], (20, (i+2)*50), cv2.FONT_HERSHEY_PLAIN, fontScale= 3, color=(0,0,0), thickness=4)


######################
prev_time = time.time()
cap = cap_setting()
model_svm_face = joblib.load('SVM.pkl')
buffer = []
MAX_SIZE = 60

##########################

while True:
    # FPS
    current_time = time.time()
    dt = current_time - prev_time
    fps = 1.0 / dt if dt > 0 else 0.0
    buffer.append(fps)
    if len(buffer) > MAX_SIZE:
        buffer.pop(0)
    prev_time = current_time
    ok, img = cap.read()
    ## From Face Recognition
    X_img_face, prediction_message_face = predict_df_generator_face(img)
    predict_message_final_face = prediction_face(model_svm_face, X_img_face, prediction_message_face)
    fps_message(img, fps)
    put_text_face(img, predict_message_final_face)
    cv2.imshow("FER", img)   # <-- missing line

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if len(buffer) > 0:
            avg = sum(buffer) / len(buffer)
            print("Average FPS:", avg)
        break

# --------------------------------------------------q
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()