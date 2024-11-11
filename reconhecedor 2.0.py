import cv2
import os
import face_recognition
import numpy as np

# Diretório onde os rostos conhecidos são armazenados
rostos_conhecidos_dir = "rostos_conhecidos"

# Função para carregar rostos conhecidos e seus embeddings
def carregar_rostos_conhecidos():
    rostos_embeddings = []
    nomes = []

    for arquivo in os.listdir(rostos_conhecidos_dir):
        if arquivo.endswith(".jpg"):
            caminho_arquivo = os.path.join(rostos_conhecidos_dir, arquivo)
            imagem = cv2.imread(caminho_arquivo)
            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

            # Encontrar os rostos e seus embeddings
            rostos_localizados = face_recognition.face_locations(imagem_rgb)
            embeddings = face_recognition.face_encodings(imagem_rgb, rostos_localizados)

            # Para cada rosto encontrado, adiciona o embedding e o nome
            for emb in embeddings:
                rostos_embeddings.append(emb)
                nomes.append(arquivo.replace(".jpg", ""))  # Usar o nome do arquivo como o nome da pessoa

    return rostos_embeddings, nomes

# Função para identificar um rosto
def identificar_rosto(imagem, rostos_embeddings, nomes):
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Encontrar os rostos na imagem
    rostos_localizados = face_recognition.face_locations(imagem_rgb)
    embeddings_detectados = face_recognition.face_encodings(imagem_rgb, rostos_localizados)

    for (top, right, bottom, left), emb in zip(rostos_localizados, embeddings_detectados):
        # Comparar o embedding detectado com os embeddings conhecidos
        distancias = face_recognition.face_distance(rostos_embeddings, emb)

        # Obter o índice do rosto mais próximo
        indice_mais_proximo = np.argmin(distancias)

        # Obter o nome da pessoa com o rosto mais próximo
        nome_identificado = nomes[indice_mais_proximo]

        # Desenhar a caixa delimitadora e o nome na imagem
        cv2.rectangle(imagem, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(imagem, nome_identificado, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return imagem

# Função para capturar vídeo e identificar rostos em tempo real
def reconhecer_rostos_em_video():
    # Carregar rostos conhecidos e seus embeddings
    rostos_embeddings, nomes = carregar_rostos_conhecidos()

    # Inicializar a câmera
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Identificar rostos no vídeo
        frame_identificado = identificar_rosto(frame, rostos_embeddings, nomes)
        
        # Mostrar a imagem com os rostos identificados
        cv2.imshow("Reconhecimento de Rostos", frame_identificado)
        
        # Encerra o programa quando pressionado 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

# Iniciar o reconhecimento de rostos
reconhecer_rostos_em_video()
