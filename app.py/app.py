from flask import Flask, render_template, redirect, url_for, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Diretório onde rostos conhecidos são armazenados
rostos_conhecidos_dir = "rostos_conhecidos"

# Níveis de acesso
niveis_de_acesso = {
    "josue": "Master",
    "eduardo": "Admin",
    "rodrigo": "Public"
}

# Funções auxiliares (carregar rostos, treinar modelo, identificar rosto)
def carregar_rostos_conhecidos():
    rostos_imagens, nomes = [], []
    for arquivo in os.listdir(rostos_conhecidos_dir):
        if arquivo.endswith(".jpg"):
            caminho_arquivo = os.path.join(rostos_conhecidos_dir, arquivo)
            imagem = cv2.imread(caminho_arquivo)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)
            resultados = face_detection.process(imagem_rgb)
            if resultados.detections:
                for rosto in resultados.detections:
                    bbox = rosto.location_data.relative_bounding_box
                    x, y, w, h = (int(bbox.xmin * imagem.shape[1]), 
                                  int(bbox.ymin * imagem.shape[0]), 
                                  int(bbox.width * imagem.shape[1]), 
                                  int(bbox.height * imagem.shape[0]))
                    rosto_cortado = cv2.resize(imagem[y:y+h, x:x+w], (160, 160))
                    rostos_imagens.append(rosto_cortado)
                    nomes.append(arquivo.replace(".jpg", ""))
    return rostos_imagens, nomes

def treinar_reconhecedor():
    rostos_imagens, nomes = carregar_rostos_conhecidos()
    rostos_imagens = np.array(rostos_imagens).reshape(len(rostos_imagens), -1)
    rosto_nomes_encoder = LabelEncoder()
    nomes_encoded = rosto_nomes_encoder.fit_transform(nomes)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(rostos_imagens, nomes_encoded)
    return knn, rosto_nomes_encoder

def identificar_rosto(knn, rosto_nomes_encoder):
    camera = cv2.VideoCapture(0)
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.2)
    
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(imagem_rgb)
        
        if resultados.detections:
            for rosto in resultados.detections:
                bbox = rosto.location_data.relative_bounding_box
                x, y, w, h = (int(bbox.xmin * frame.shape[1]), 
                              int(bbox.ymin * frame.shape[0]), 
                              int(bbox.width * frame.shape[1]), 
                              int(bbox.height * frame.shape[0]))
                rosto_cortado = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
                predicao = knn.predict(rosto_cortado.reshape(1, -1))
                nome_identificado = rosto_nomes_encoder.inverse_transform(predicao)[0]
                camera.release()
                return nome_identificado
        if cv2.waitKey(1) & 0xFF == 27:
            break
    camera.release()
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    knn, rosto_nomes_encoder = treinar_reconhecedor()
    usuario = identificar_rosto(knn, rosto_nomes_encoder)
    if usuario:
        nivel_acesso = niveis_de_acesso.get(usuario.lower(), "Desconhecido")
        return jsonify({"usuario": usuario, "nivel": nivel_acesso})
    return jsonify({"erro": "Usuário não reconhecido"})

if __name__ == '__main__':
    app.run(debug=True)
