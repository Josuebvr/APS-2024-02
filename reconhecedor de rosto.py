import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Inicializando o MediaPipe para detecção de rostos
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Diretório onde os rostos conhecidos são armazenados
rostos_conhecidos_dir = "rostos_conhecidos"

# Função para carregar rostos conhecidos e os nomes associados
def carregar_rostos_conhecidos():
    rostos_imagens = []
    nomes = []
    
    for arquivo in os.listdir(rostos_conhecidos_dir):
        if arquivo.endswith(".jpg"):
            caminho_arquivo = os.path.join(rostos_conhecidos_dir, arquivo)
            imagem = cv2.imread(caminho_arquivo)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            resultados = face_detection.process(imagem_rgb)
            
            if resultados.detections:
                for rosto in resultados.detections:
                    bbox = rosto.location_data.relative_bounding_box
                    x, y, w, h = int(bbox.xmin * imagem.shape[1]), int(bbox.ymin * imagem.shape[0]), \
                                 int(bbox.width * imagem.shape[1]), int(bbox.height * imagem.shape[0])
                    rosto_cortado = imagem[y:y+h, x:x+w]
                    rostos_imagens.append(rosto_cortado)
                    nomes.append(arquivo.replace(".jpg", ""))  # Usar o nome do arquivo como o nome da pessoa
    
    return rostos_imagens, nomes

# Função para treinar o classificador
def treinar_reconhecedor(rostos_imagens, nomes):
    rostos_imagens = [cv2.resize(rosto, (160, 160)) for rosto in rostos_imagens]
    rostos_imagens = np.array(rostos_imagens)
    
    # Usando KNN (K-Nearest Neighbors) para reconhecer rostos
    rosto_nomes_encoder = LabelEncoder()
    nomes_encoded = rosto_nomes_encoder.fit_transform(nomes)
    
    # Usando KNeighborsClassifier para treinamento
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(rostos_imagens.reshape(len(rostos_imagens), -1), nomes_encoded)
    
    return knn, rosto_nomes_encoder

# Função para identificar um rosto
def identificar_rosto(imagem, knn, rosto_nomes_encoder):
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultados = face_detection.process(imagem_rgb)
    
    if resultados.detections:
        for rosto in resultados.detections:
            bbox = rosto.location_data.relative_bounding_box
            x, y, w, h = int(bbox.xmin * imagem.shape[1]), int(bbox.ymin * imagem.shape[0]), \
                         int(bbox.width * imagem.shape[1]), int(bbox.height * imagem.shape[0])
            rosto_cortado = imagem[y:y+h, x:x+w]
            rosto_cortado = cv2.resize(rosto_cortado, (160, 160))
            
            # Reshaping para o formato que o modelo espera
            rosto_cortado = rosto_cortado.reshape(1, -1)
            
            # Prever a identidade do rosto
            predicao = knn.predict(rosto_cortado)
            nome_identificado = rosto_nomes_encoder.inverse_transform(predicao)
            
            # Desenhar o nome do rosto detectado na imagem
            cv2.putText(imagem, nome_identificado[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return imagem

# Função para capturar vídeo e identificar rostos em tempo real
def reconhecer_rostos_em_video():
    # Carregar rostos conhecidos e treinar o modelo
    rostos_imagens, nomes = carregar_rostos_conhecidos()
    knn, rosto_nomes_encoder = treinar_reconhecedor(rostos_imagens, nomes)
    
    # Inicializar a câmera
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Identificar rostos no vídeo
        frame_identificado = identificar_rosto(frame, knn, rosto_nomes_encoder)
        
        # Mostrar a imagem com os rostos identificados
        cv2.imshow("Reconhecimento de Rostos", frame_identificado)
        
        # Encerra o programa quando pressionado 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

# Iniciar o reconhecimento de rostos
reconhecer_rostos_em_video()
