# Nombre: Esmerlin Severino Paredes 
# Matricula: 24-EISN-2-033

# Para detectar personas en tiempo real usando el modelo Yolov8n.pt 
from ultralytics import YOLO
import cv2 # Para trabajar con la camara y mostrar los resultados 

# Para cargar el modelo preentrenado de yolov8
Modelo = YOLO("yolov8n.pt")

# Para activar la cámara 0 = cámara principal
Camara = cv2.VideoCapture(0)

# Bucle para procesar cada frame de la camara 
while True:
    Ret, Frame = Camara.read() # Para leer un frame de la camara 

    if not Ret: # Para verificar si se pudo abrir la camara 
        print("Error al abrir la cámara")
        break

    #  Para procesar el frame con el modelo y obtener los resultados de detenccion 
    Resultados = Modelo(Frame)

    # Para dibujar las cajas de deteccion en el frame 
    for Resultado in Resultados:
        Cajas = Resultado.boxes # Para obtener las cajas de deteccion del resultado
 
        for Caja in Cajas: # Para verificar si la clase detectada es una persona 
            Clase = int(Caja.cls[0])

            # 0 = persona en YOLO
            if Clase == 0:
                x1, y1, x2, y2 = map(int, Caja.xyxy[0]) # Para obtener las coordenadas de la caja de deteccion 

                #  Para Dibujar rectángulo
                cv2.rectangle(Frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Para poner el texto persona sobre la caja de deteccion 
                cv2.putText(Frame, "Persona", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #  Para mostrar la cámara
    cv2.imshow("Deteccion de Personas", Frame)

    # Para Salir con tecla ESC
    if cv2.waitKey(1) == 27:
        break

Camara.release() # Para liberar la camara 
cv2.destroyAllWindows() # para cerrar las ventanas de OpenCV