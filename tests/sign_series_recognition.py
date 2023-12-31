import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp



COUNT = 7
class SignSeriesRecognition:
    def __init__(self, start_sign, end_sign):
        self.START_SIGN = start_sign
        self.END_SIGN = end_sign
        self.is_working = False
        self.sign_series = []
    def begin(self):
        self.is_working = True
        self.sign_series = []
    def add_sign(self, sign):
        if self.is_working:
            if len(self.sign_series) == 0:
                self.sign_series.append(sign)
            if self.sign_series[-1] != sign:
                self.sign_series.append(sign)
    def get_sign_series(self):
        return [i+1 for i in self.sign_series]
    
    def end(self):
        self.is_working = False
        if self.is_working:
            self.sign_series = []



if __name__ == "__main__":

    recognizer = SignSeriesRecognition(4, 3)    

    i1 = tf.lite.Interpreter(model_path='./lite/model1.tflite')
    s1 = i1.get_signature_runner()

    i2 = tf.lite.Interpreter(model_path='./lite/model2.tflite')
    s2 = i2.get_signature_runner()

    i3 = tf.lite.Interpreter(model_path='./lite/model3.tflite')
    s3 = i3.get_signature_runner()


    superInterpreter = tf.lite.Interpreter(model_path='./lite/supermodel.tflite')
    superSignature = superInterpreter.get_signature_runner()

    language = [i + 1 for i in range(13)]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    cap = cv2.VideoCapture(0)

    predicted_sign_index = -1
    
    currentSignCount = 0
    lastSign = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        sureness = 0
        
        if results.multi_hand_world_landmarks:
            for hand_landmarks in results.multi_hand_world_landmarks:
                matrix = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark], dtype=np.float32)
         
                p1 = s1(flatten_input=matrix)
                p1 = p1[list(p1.keys())[0]]
                p1[0] = np.exp(p1[0])/sum(np.exp(p1[0])) #softmax
         
            
                p2 = s2(flatten_input=matrix)
                p2 = p2[list(p2.keys())[0]]
                p2[0] = np.exp(p2[0])/sum(np.exp(p2[0])) #softmax
         
         
                p3 = s3(flatten_input=matrix)
                p3 = p3[list(p3.keys())[0]]
                p3[0] = np.exp(p3[0])/sum(np.exp(p3[0])) #softmax

                x = np.array([p1[0], p2[0], p3[0]], dtype=np.float32)

                superP = superSignature(flatten_input=x)
                superP = superP[list(superP.keys())[0]]
                superP[0] = np.exp(superP[0])/sum(np.exp(superP[0])) #softmax

                index = np.argmax(superP[0]) 
                sureness = superP[0][predicted_sign_index]
                    
                if lastSign != predicted_sign_index:
                    lastSign = predicted_sign_index
                    currentSignCount = 0
                else:
                    currentSignCount += 1

                if currentSignCount == COUNT:
                    currentSignCount = 0
                    predicted_sign_index  = index


                if recognizer.is_working and sureness > 0.9:
                    if predicted_sign_index == recognizer.END_SIGN:
                        print("ROZPOZNANO GEST KONCA")
                        print(recognizer.get_sign_series())
                        recognizer.end()
                    else: 
                        if predicted_sign_index != recognizer.START_SIGN :
                            recognizer.add_sign(predicted_sign_index)
                        
                if predicted_sign_index == recognizer.START_SIGN and not recognizer.is_working and sureness > 0.9:
                    print("ROZPOZNANO GEST STARTU")
                    recognizer.begin()  
        else :
            predicted_sign_index = -1

        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (0, 0), (100, 80), (0, 0, 255), -1)
        cv2.putText(frame, str(language[predicted_sign_index]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('testowanie', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()