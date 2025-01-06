# Tarea evaluable CE_5073 3.1
Clasificador de pinguinos a partir del dataset de Palmer para entrenar y desplegar modelos de clasificación.

## Modelos
1. Regresión Logística
2. SVM
3. Árboles de Decisión
4. KNN

## Estructura
- `models/`: Contiene los modelos serializados.
- `server/`: Contiene los servidores Flask para los modelos.
- `client/`: Cliente que envía peticiones a los servidores.
- `notebooks/`: Notebooks para entrenar los modelos.

## Uso
1. Entrena los modelos desde los notebooks.
2. Ejecuta los servidores en diferentes terminales.
3. Ejecuta el cliente para enviar peticiones.

## Requisitos
Instala las dependencias con: 
pip install -r requirements.txt