# Nombre: Esmerlin Severino Paredes
# Matrícula: 24-EISN-2-033

import cv2
import gradio as gr # Importe gradio para la interfaz con el usuario 
import time
from ultralytics import YOLO
import pygame  # Importe pygame para manejar el sonido de la musica de fondo 

# Para Cargar el modelo
Modelo = YOLO("yolov8n.pt")

# Para inicializar el y cargar el sonido de fondo 
pygame.mixer.init()
pygame.mixer.music.load("sonido.mp3") 

# Variable global para controlar el estado de la camara 
corriendo = False
CamaraGlobal = None  # Para  almacenar la instancia de la camara seleccionada 
TipoCamara = None  #  Para almacenar el tipo de camara seleccionada pc o iriun

# Funsion para seleccionar la camara de la pc y actualizar el estado de la interfaz
def UsarCamaraPC():
    global TipoCamara
    TipoCamara = "pc"
    return None, "💻 Cámara de la PC seleccionada"

# Funsion para seleccionar la camara de iriun y actualizar el estado de la interfaz
def UsarCamaraIriun():
    global TipoCamara
    TipoCamara = "iriun"
    return None, "📱 Cámara Iriun seleccionada"

def IniciarCamara(): # Funsion para iniciar la camara y procesar los frames
    global corriendo, CamaraGlobal, TipoCamara
    corriendo = True
    
    # selección manual de cámara
    if TipoCamara == "pc":
        CamaraGlobal = cv2.VideoCapture(0)

    elif TipoCamara == "iriun":
        for i in [1, 2, 3]:
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            if ret:
                print(f"Iriun detectada en indice: {i}")
                CamaraGlobal = cap
                break
            cap.release()
    
    else:
        yield None, "❌ Selecciona una cámara primero"
        return

    if CamaraGlobal is None:
        yield None, "❌ Error: No se encontró la cámara"
        return
    
    ultimo_tiempo = time.time() # Variable para almacenar el ultimo tiempo que se detecto una persona 
    tiempo_limite = 5
    musica_sonando = False  # Variable para controlar si la musica esta sonando o no 

# Bucle para procesar cada frame de la camara 
    while corriendo:
        ret, frame = CamaraGlobal.read()
        if not ret:
            break

        resultados = Modelo(frame) # Para detectar objetos en el frame 
        estado = "Persona no detectada, música detenida"
        persona_detectada = False

        for r in resultados: # Para verificar cada estado de deteccion 
            for caja in r.boxes:
                clase = int(caja.cls[0])
                if Modelo.names[clase] == "person":
                    persona_detectada = True
                    estado = "Música activa 🎶"
                    x1, y1, x2, y2 = map(int, caja.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tiempo_actual = time.time() # Para obtener el tiempo actual sin detectar personas 
        
        if persona_detectada:
            ultimo_tiempo = tiempo_actual

            # Para iniciar la musica si no esta sonando y se detecta una persona 
            if not musica_sonando:
                pygame.mixer.music.play(-1)
                musica_sonando = True

        else:
            tiempo_pasado = int(tiempo_actual - ultimo_tiempo)

            if tiempo_pasado < tiempo_limite:
                cuenta = tiempo_limite - tiempo_pasado
                estado = f"Deteniendo música en: {cuenta} ⏳"
            else:
                estado = "Persona no detectada, música detenida 🔇"

            # Para detener la musica despues de 5 segundos
            if tiempo_actual - ultimo_tiempo >= 5:
                if musica_sonando:
                    pygame.mixer.music.stop()
                    musica_sonando = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb, estado

    CamaraGlobal.release()
    yield None, "Cámara detenida"

def detener_camara():
    global corriendo, CamaraGlobal
    corriendo = False
    
    if CamaraGlobal is not None:
        CamaraGlobal.release()
    
    pygame.mixer.music.stop()  # Para detener la musica al apagar la camara 
    
    return None, "Cámara apagada"

# CSS personalizado para mejorar la apariencia de la interfaz 
custom_css = """
.gradio-container {
    background: linear-gradient(180deg, #FFFFFF 0%, #D0E1F9 100%) !important;
}
/* CAMBIO: Título en Times New Roman */
h1 {
    font-family: 'Times New Roman', Times, serif !important;
    color: #1A3A5F !important;
}
h3, p {
    color: #1A3A5F !important;
}
.btn-inicio {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    color: white !important;
    border: none !important;
}
.btn-detener {
    background: linear-gradient(90deg, #f85032 0%, #e73827 100%) !important;
    color: white !important;
    border: none !important;
}
"""

with gr.Blocks(css=custom_css) as interfaz:
    
    gr.Markdown( # Para mostrar el título de la aplicación
        """
        # Sistema Inteligente Para Negocios 🏪
        ### Sistema Inteligente de Detección de Clientes en Tiempo Real
        #### Presiona el botón para iniciar la cámara
        """
    )

    with gr.Row(): # Para organizar los botones en una fila 
        BotonInicio = gr.Button("▶ Iniciar Cámara", elem_classes=["btn-inicio"])
        BotonDetener = gr.Button("⏹ Detener Cámara", elem_classes=["btn-detener"])
        BotonPC = gr.Button("💻 Usar Webcam")  
        BotonIriun = gr.Button("📱 Usar Iriun")  

    ImagenSalida = gr.Image(label="Cámara en vivo")
    EstadoSalida = gr.Textbox(label="Estado del sistema")

    BotonInicio.click( # Boton para iniciar la camara y actualizar el estado 
        fn=IniciarCamara,
        inputs=[],
        outputs=[ImagenSalida, EstadoSalida]
    )
    
    BotonDetener.click( # boton para detener la camara y actualizar el estado 
        fn=detener_camara,
        inputs=[],
        outputs=[ImagenSalida, EstadoSalida]
    )

    # botones para seleccionar la camara y actualizar el estado de la interfaz 
    BotonPC.click(
        fn=UsarCamaraPC,
        inputs=[],
        outputs=[ImagenSalida, EstadoSalida]
    )

    BotonIriun.click(
        fn=UsarCamaraIriun,
        inputs=[],
        outputs=[ImagenSalida, EstadoSalida]
    )

if __name__ == "__main__": # Para ejecutar la interfaz 
    interfaz.launch(theme=gr.themes.Soft()) # Para lanzar la interfaz con un tema suave 