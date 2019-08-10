# usher-rec (core del sistema Usher)
Reconocimiento de ocupación de asientos o ubicaciones mediante el reconocimiento de imágenes de video de cámaras.
### Funciones
Reconocimiento mediante deep learning y machine vision de ubicaciones libres y ocupadas a partir del video de una cámara

Registro del estado de ocupación de las distintas ubicaciones

Operación en background (simil servicio)

### Características de flexibilidad
Multiplataforma (desarrollado en python)

Configuración remota vía BBDD

Desarrollado en conjunto a API (usher-api) para su control

### Características de escalabilidad
Operación con una o múltiples cámaras

Operación en simultáneo a otros CamServer

Asociación de ubicaciones coincidentes en múltiples cámaras

Priorización de estados indicados por cada CamServer (prioridad)

Ponderación de información de Cámaras por cada ubicación (pesos)

### Características de robustez
Máquina de estados coordinada con usher-api vía BBDD

Inicio, reinicio, suspensión remota vía BBDD

Recuperación automática de estado al ejecutar CamServer

Recuperación automática ante falla de conexión con BBDD

Recuperación automática ante falla de comunicación con cámaras

Generación automática de configuración CamServer si no existiera

Generación automática de registro de estado de ubicaciones si no existiera

Control de temporización para:
- Lectura de estado en BBDD
- Grabación de estado en BBDD
- Evaluación de ocupación de ubicaciones
- Frecuencia de captura de frames de video
- Comprobación de conexión a Cámaras post falla
