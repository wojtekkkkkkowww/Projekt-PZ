import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_signs = 8
number_of_frames = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_signs):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Zbieranie danych do znaku {}'.format(j+1))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Nacisnij f zeby zaczac', (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('f'):
            break

    c = 0
    while c < number_of_frames:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(c)), frame)
        c += 1

cap.release()
cv2.destroyAllWindows()