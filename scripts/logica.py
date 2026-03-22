# Nombre: Esmerlin Severino Paredes
# Matrícula: 24-EISN-2-033

from ultralytics import YOLO
import cv2
import time # Para medir el tiempo sin detectar personas 

# Para cargar modelo
Modelo = YOLO("yolov8n.pt")

# Para activar la camara 
Camara = cv2.VideoCapture(0)


TiempoSinPersona = 0 # Variable para contar el tiempo sin detectar personas 
TiempoInicio = None # Variable para almacenar el tiempo de inicio sin detectar personas 
EstadoLuz = "APAGADA" # Variable para almacenar el estado de la luz 

while True: # Bucle para procesar cada frame de la camara 
    Ret, Frame = Camara.read()
    
    if not Ret: # para verificar si se pudo abrir la camara 
        print("Error al abrir la cámara")
        break

    # Detectar objetos en el frame 
    Resultados = Modelo(Frame)

    PersonaDetectada = False # variable para indicar si se detecto una persona en el frame 

    for R in Resultados: # Para verificar cada resultado de detenccion 
        for Caja in R.boxes: # Para verificar cada caja de deteccion 
            Clase = int(Caja.cls[0]) # Pata obtener la clase detectada en la caja 
            NombreClase = Modelo.names[Clase] # para obtener el nombre de la clase 

            if NombreClase == "person":
                PersonaDetectada = True

                # Para Dibujar el cuadro
                X1, Y1, X2, Y2 = map(int, Caja.xyxy[0])
                cv2.rectangle(Frame, (X1, Y1), (X2, Y2), (0,255,0), 2)
                cv2.putText(Frame, "Persona", (X1, Y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # para controlar la luz dependiendo de si esta detectando persona o no 
    if PersonaDetectada:
        print("Persona detectada → Luz encendida")

        EstadoLuz = "ENCENDIDA"
        TiempoInicio = None  # reinicia contador

    else: # si no se detecta persona 
        print("Sin persona → contando...")

        if TiempoInicio is None: # Para iniciar el contador si no se ha iniciado 
            TiempoInicio = time.time()

        TiempoSinPersona = time.time() - TiempoInicio # Para calcular el tiempo sin detectar personas

        if TiempoSinPersona > 5: # Si han pasado mas de 5 segundos sin detectar personas, apagar la luz 
            if EstadoLuz != "APAGADA":
                print("Luz apagada")
                EstadoLuz = "APAGADA"

    #  Para Mostrar la cámara
    cv2.imshow("Sistema Inteligente", Frame)

    if cv2.waitKey(1) & 0xFF == 27: # para salir con tecla ESC
        break

Camara.release() # Para liberar la camara 
cv2.destroyAllWindows() # para cerrar las ventanas de OpenCV