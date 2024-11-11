import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Inicializando MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Diretório onde os rostos conhecidos são armazenados
rostos_conhecidos_dir = "rostos_conhecidos"

# Dicionário de níveis de acesso
niveis_de_acesso = {
    "josue": "Master",
    "eduardo": "Admin",
    "rodrigo": "Public"
}

def carregar_rostos_conhecidos():
    rostos_imagens, nomes = [], []
    for arquivo in os.listdir(rostos_conhecidos_dir):
        if arquivo.endswith(".jpg"):
            caminho_arquivo = os.path.join(rostos_conhecidos_dir, arquivo)
            imagem = cv2.imread(caminho_arquivo)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            resultados = face_detection.process(imagem_rgb)
            if resultados.detections:
                for rosto in resultados.detections:
                    bbox = rosto.location_data.relative_bounding_box
                    x, y, w, h = (int(bbox.xmin * imagem.shape[1]), 
                                  int(bbox.ymin * imagem.shape[0]), 
                                  int(bbox.width * imagem.shape[1]), 
                                  int(bbox.height * imagem.shape[0]))
                    rosto_cortado = imagem[y:y+h, x:x+w]
                    rostos_imagens.append(cv2.resize(rosto_cortado, (160, 160)))
                    nomes.append(arquivo.replace(".jpg", ""))
    return rostos_imagens, nomes

def treinar_reconhecedor(rostos_imagens, nomes):
    rostos_imagens = np.array(rostos_imagens).reshape(len(rostos_imagens), -1)
    rosto_nomes_encoder = LabelEncoder()
    nomes_encoded = rosto_nomes_encoder.fit_transform(nomes)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(rostos_imagens, nomes_encoded)
    return knn, rosto_nomes_encoder

def identificar_rosto(imagem, knn, rosto_nomes_encoder):
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultados = face_detection.process(imagem_rgb)
    if resultados.detections:
        for rosto in resultados.detections:
            bbox = rosto.location_data.relative_bounding_box
            x, y, w, h = (int(bbox.xmin * imagem.shape[1]), 
                          int(bbox.ymin * imagem.shape[0]), 
                          int(bbox.width * imagem.shape[1]), 
                          int(bbox.height * imagem.shape[0]))
            rosto_cortado = cv2.resize(imagem[y:y+h, x:x+w], (160, 160))
            predicao = knn.predict(rosto_cortado.reshape(1, -1))
            nome_identificado = rosto_nomes_encoder.inverse_transform(predicao)[0]
            nivel_acesso = niveis_de_acesso.get(nome_identificado.lower(), "Desconhecido")
            cv2.putText(imagem, f"{nome_identificado} - {nivel_acesso}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return imagem

def reconhecer_rostos_em_video():
    rostos_imagens, nomes = carregar_rostos_conhecidos()
    knn, rosto_nomes_encoder = treinar_reconhecedor(rostos_imagens, nomes)
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame_identificado = identificar_rosto(frame, knn, rosto_nomes_encoder)
        cv2.imshow("Reconhecimento de Rostos - Acesso", frame_identificado)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    camera.release()
    cv2.destroyAllWindows()

# Iniciar reconhecimento com níveis de acesso
reconhecer_rostos_em_video()
