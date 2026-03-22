# Nombre: Esmerlin Severino Paredes 
# Matricula: 24-EISN-2-033

# Para probar la carga del modelo Yolov8n.pt
from ultralytics import YOLO 
# Para Cargar el modelo Yolov8n.pt 
model = YOLO("yolov8n.pt")
# Para verificar que el modelo se ha cargado corectamente 
print("Modelo cargado correctamente")