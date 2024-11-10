import cv2
import mediapipe as mp 

camera = cv2.VideoCapture(0)
reconhecimeto_rosto = mp.solutions.face_detection
reconhecedor_rostos = reconhecimeto_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
    verificador, frame = camera.read()

    if not verificador:
        break
    
    lista_rostos = reconhecedor_rostos.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)

    cv2.imshow("Rostos na Webcam", frame)

    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()