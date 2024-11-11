import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import ttk

# Inicializando a webcam
camera = cv2.VideoCapture(0)

# MediaPipe para detecção de rostos
reconhecimeto_rosto = mp.solutions.face_detection
reconhecedor_rostos = reconhecimeto_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

# Diretório para salvar os rostos conhecidos
rostos_conhecidos_dir = "rostos_conhecidos"

# Verificar se o diretório de rostos existe; caso contrário, criar
if not os.path.exists(rostos_conhecidos_dir):
    os.makedirs(rostos_conhecidos_dir)

# Função para salvar rosto capturado
def salvar_rosto(identidade, frame, rosto_bbox):
    x, y, w, h = int(rosto_bbox.xmin * frame.shape[1]), int(rosto_bbox.ymin * frame.shape[0]), \
                 int(rosto_bbox.width * frame.shape[1]), int(rosto_bbox.height * frame.shape[0])
    rosto = frame[y:y+h, x:x+w]
    caminho_arquivo = os.path.join(rostos_conhecidos_dir, f"{identidade}.jpg")
    cv2.imwrite(caminho_arquivo, rosto)
    print(f"Rosto salvo como {caminho_arquivo}!")

# Função para capturar o rosto no momento atual
def capturar_rosto(nome):
    global frame_atual
    if nome and frame_atual is not None:
        # Realiza a detecção e captura o rosto na imagem atual
        frame_rgb = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2RGB)  # Conversão correta para RGB
        lista_rostos = reconhecedor_rostos.process(frame_rgb)
        
        if lista_rostos.detections:
            for rosto in lista_rostos.detections:
                desenho.draw_detection(frame_atual, rosto)
                bbox = rosto.location_data.relative_bounding_box
                salvar_rosto(nome, frame_atual, bbox)

# Configuração da interface Tkinter
root = tk.Tk()
root.title("Cadastro de Rostos")
root.geometry("800x600")

nome_var = tk.StringVar()

def encerrar_programa(event=None):
    global running
    running = False
    root.quit()

def atualizar_feed_video():
    global frame_atual
    verificador, frame_atual = camera.read()
    if verificador:
        # Mostrar o feed da câmera na interface Tkinter
        # Não é necessário converter para RGB aqui, pois a captura da webcam é em BGR, que é o padrão do OpenCV
        frame_tk = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2RGB)  # Converte para RGB para exibição no Tkinter
        frame_tk = cv2.resize(frame_tk, (640, 480))  # Redimensiona a imagem para caber na interface
        img = tk.PhotoImage(data=cv2.imencode('.png', frame_tk)[1].tobytes())  # Cria a imagem no formato Tkinter
        lbl_video.imgtk = img
        lbl_video.configure(image=img)

# Função para capturar nome e salvar foto (após pressionar "Enter")
def confirmar_nome(event=None):
    nome = nome_var.get().strip()
    if nome:
        # Captura o rosto no momento atual
        capturar_rosto(nome)
        nome_var.set("")  # Limpa o campo de texto após salvar

# Configurar evento de pressionar "Enter" para confirmar nome
root.bind('<Return>', confirmar_nome)

# Widgets da interface
lbl_video = ttk.Label(root)
lbl_video.pack(side="left", padx=10, pady=10)

frm_controles = ttk.Frame(root)
frm_controles.pack(side="right", fill="y", padx=10, pady=10)

ttk.Label(frm_controles, text="Digite o nome completo:").pack(pady=5)
entry_nome = ttk.Entry(frm_controles, textvariable=nome_var)
entry_nome.pack(pady=5)
btn_sair = ttk.Button(frm_controles, text="Encerrar", command=encerrar_programa)
btn_sair.pack(pady=20)

# Vincular tecla ESC para encerrar
root.bind('<Escape>', encerrar_programa)

running = True
while running:
    atualizar_feed_video()
    root.update()

camera.release()
cv2.destroyAllWindows()
