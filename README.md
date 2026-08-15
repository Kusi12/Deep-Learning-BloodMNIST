# Clasificación de Células Sanguíneas con ResNet (BloodMNIST)


Trabajo final del seminario **Deep Learning applied in Bioimage Analysis** -- Pontificia Universidad Católica del Perú (PUCP).

**Autores:** Gianfranco Moises Poma Canchari --100%, Valeshka Lavado Guerra --100%, Andy Edery Wusen --100%, Kusi Uñapillco Franco --100%

---

## Tabla de contenidos

- [Descripción general](#-descripción-general)
- [Dataset: BloodMNIST](#-dataset-bloodmnist)
- [Arquitectura del modelo](#-arquitectura-del-modelo)
- [Requisitos](#-requisitos)
- [Estructura del notebook](#-estructura-del-notebook)
- [Cómo ejecutar](#-cómo-ejecutar)
- [Resultados](#-resultados)
- [Pruebas interactivas](#-pruebas-interactivas)
- [Limitaciones y advertencias](#-limitaciones-y-advertencias)
- [Archivos generados](#-archivos-generados)

---

## Descripción general

El notebook `Blood_classification_v4.ipynb` implementa un pipeline completo de *deep learning* con **PyTorch** para resolver un problema de **clasificación multiclase de células sanguíneas periféricas** a partir de imágenes microscópicas de baja resolución. El objetivo es evaluar la capacidad de abstracción y generalización de una arquitectura convolucional profunda (ResNet) entrenada **desde cero** (sin pesos preentrenados) sobre el benchmark **BloodMNIST**.

## Relevancia

La aplicación biomédica de esta tarea es automatizar el conteo y clasificación celular en frotis de sangre periférica, lo cual acelera diagnósticos en hematología y reduce la carga operativa en laboratorios.

## Dataset: BloodMNIST

Se utiliza **BloodMNIST**, parte de la colección estandarizada **MedMNIST v2**, un conjunto de benchmarks ligeros para evaluar algoritmos de Machine Learning en imágenes biomédicas.

En adición, as imágenes originales tenían una resolución de 3x360x363 píxeles. Luego, fueron recortadas en el centro a 3x200x200 y finalmente redimensionadas a 3x28x28 mediante interpolación spline cúbica.

| Propiedad | Valor |
|---|---|
| Total de imágenes | 17,092 |
| Número de clases | 8 |
| Resolución | 3×28×28 (RGB) |
| Origen | Células individuales normales, sin infecciones, enfermedades hematológicas/oncológicas ni tratamiento farmacológico |
| Preprocesamiento | Recorte central + redimensionado a 28×28 px |

**Clases (8):** `Basophil`, `Eosinophil`, `Erythroblast`, `IG` (granulocito inmaduro), `Lymphocyte`, `Monocyte`, `Neutrophil`, `Platelet`.

Los datos se descargan automáticamente mediante la librería `medmnist` (`train`, `val`, `test`) y se cargan en `DataLoader`s de PyTorch con `batch_size=128`.

En adición, se mantuvo la división oficial de MedMNIST (70% train, 10% val, 20% test) para garantizar una comparación justa y reproducible con el benchmark oficial.

## Arquitectura del modelo

Se implementa **ResNet-18** (`torchvision.models.resnet18`, `weights=None`) con dos modificaciones clave para adaptarla a imágenes pequeñas de 28×28 px:

- **`conv1`** se reemplaza por una convolución 3×3, stride 1, padding 1 (en vez de la 7×7/stride 2 original), evitando una reducción espacial agresiva desde el inicio.
- **`maxpool`** inicial se reemplaza por `nn.Identity()` (se elimina).
- **Capa final (`fc`)** adaptada a 8 clases de salida.

### Hiperparámetros de entrenamiento

| Parámetro | Valor |
|---|---|
| Épocas | 100 |
| Función de pérdida | Cross-Entropy Loss |
| Optimizador | Adam (lr = 0.001) |
| Batch size | 128 |
| Scheduler | `MultiStepLR`, milestones en épocas 50 y 75, gamma = 0.1 |
| Hardware recomendado | GPU (Colab T4) |

La elección de Adam (lr=0.001), Batch Size de 128, Cross-Entropy Loss, y el uso de un scheduler (reduciendo la tasa de aprendizaje por 0.1 en las épocas 50 y 75 durante 100 épocas) no es arbitraria; replica exactamente el esquema base (baseline) oficial establecido por los autores del benchmark MedMNIST v2.

Durante el entrenamiento se guarda el mejor modelo (mayor precisión de validación) en `mejor_modelo_bloodmnist.pth`, y se registra el historial de *loss* de entrenamiento y precisión de validación por época para graficar las curvas de trazabilidad.

## Requisitos

El notebook está pensado para ejecutarse en **Google Colab** (incluye badge de apertura directa y usa `google.colab.files` para la carga interactiva de imágenes). Dependencias principales:

```
torch
torchvision
medmnist
numpy
matplotlib
seaborn
scikit-learn
Pillow
```

La única instalación manual necesaria es `medmnist` (celda 1); el resto viene preinstalado en el entorno de Colab.

## Estructura del notebook

El notebook consta de 11 celdas organizadas en un flujo secuencial:

1. **Introducción (markdown)** — Contexto del trabajo, descripción del dataset y metodología.
2. **Celda 1 — Instalación:** `pip install medmnist`.
3. **Celda 2 — Datos:** imports, transformaciones (`ToTensor` + `Normalize`) y creación de los `DataLoader` de train/val/test.
4. **Celda 3 — Modelo:** construcción y adaptación de ResNet-18, envío a GPU/CPU según disponibilidad.
5. **Celda 4 — Entrenamiento:** bucle de 100 épocas con validación por época, *scheduler*, guardado del mejor checkpoint y gráficos de *loss*/precisión.
6. **Celda 5 — Evaluación en test:** carga del mejor modelo, cálculo de precisión final, `classification_report` y matriz de confusión (heatmap).
7. **Celda 6 (markdown) — Prueba interactiva:** instrucciones para subir una imagen propia, con advertencia sobre la baja resolución del dataset y que **no está pensado para uso clínico**.
8. **Celda 6 — Código:** carga de imagen subida por el usuario, preprocesamiento, inferencia y visualización de la predicción con nivel de confianza.
9. **Celda 7 (markdown) — Validación aleatoria:** explica la evaluación con una muestra aleatoria del propio conjunto de test (`.npz`).
10. **Celda 7 — Código:** selecciona una imagen de test al azar, la clasifica, compara contra la etiqueta real (verde = acierto, rojo = error) y muestra el top-3 de probabilidades.

## Cómo ejecutar

1. Abrir el notebook en Google Colab (botón "Open in Colab" al inicio).
2. Activar el entorno de ejecución con GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)`.
3. Ejecutar las celdas en orden desde la instalación de `medmnist` hasta la evaluación en test.
4. (Opcional) Ejecutar la celda de prueba interactiva para subir una imagen propia y clasificarla.
5. (Opcional) Ejecutar la celda de validación aleatoria tantas veces como se desee para inspeccionar predicciones individuales del conjunto de test.

> El entrenamiento completo (100 épocas) puede tardar un tiempo considerable incluso con GPU T4; el mejor checkpoint se guarda automáticamente en disco durante el proceso.

## Resultados

Resultados obtenidos en la corrida registrada en el notebook:

- **Mejor precisión de validación:** 95.91%
- **Precisión final en test:** 94.94%

**Reporte de clasificación (test set, 3,421 imágenes):**

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Basophil | 0.93 | 0.92 | 0.92 | 244 |
| Eosinophil | 1.00 | 0.98 | 0.99 | 624 |
| Erythroblast | 0.97 | 0.95 | 0.96 | 311 |
| IG | 0.89 | 0.87 | 0.88 | 579 |
| Lymphocyte | 0.95 | 0.95 | 0.95 | 243 |
| Monocyte | 0.86 | 0.92 | 0.89 | 284 |
| Neutrophil | 0.96 | 0.97 | 0.97 | 666 |
| Platelet | 1.00 | 1.00 | 1.00 | 470 |
| **Accuracy global** | | | **0.95** | 3421 |
| Macro avg | 0.94 | 0.95 | 0.94 | 3421 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 3421 |

El notebook también genera:
- Curvas de *loss* de entrenamiento y precisión de validación por época (con líneas verticales marcando las caídas del learning rate en las épocas 50 y 75).
- Matriz de confusión (heatmap) sobre el conjunto de test.

Las clases con menor desempeño son `IG` (granulocito inmaduro) y `Monocyte`, probablemente por mayor solapamiento visual con otras clases mieloides a resolución tan baja.

## Pruebas interactivas

El notebook incluye dos mecanismos para probar el modelo entrenado más allá de las métricas agregadas:

1. **Subida de imagen propia:** permite cargar cualquier imagen `.jpg`/`.png` de una célula sanguínea, redimensionarla a 28×28 px y obtener la clase predicha junto con su nivel de confianza (softmax).
2. **Muestra aleatoria del test set oficial:** extrae una imagen al azar directamente del archivo `bloodmnist.npz`, la clasifica y compara contra su etiqueta real, mostrando también el top-3 de probabilidades.

## Limitaciones y advertencias

Restricción de resolución y uso clínico: Los conjuntos de datos de MedMNIST están diseñados exclusivamente para benchmarking ligero de modelos. Las imágenes de BloodMNIST están comprimidas a 28x28  píxeles, lo que genera una pérdida masiva de detalle frente a las fotografías reales de laboratorio (3x360x363 originales recortadas a 3x200x200). Por esta razón, los autores advierten explícitamente que el modelo no está destinado para uso clínico, ya que una reducción tan sustancial de la resolución puede ser insuficiente para capturar patologías complejas.

Entrenamiento desde cero: El modelo se entrena completamente desde cero (sin pesos preentrenados de transfer learning), lo que limita la capacidad de generalización frente a variaciones en la tinción o artefactos de captura externos.

## Propuestas de mejora para traducción e investigación futura

Para superar estas limitaciones y avanzar hacia un entorno de validación real o clínico, se propone:

1. Uso de imágenes de alta resolución: Entrenar arquitecturas con frotis de sangre periférica completos y de alta resolución que no omitan detalles ultraestructurales clave de las células.

2. Transfer Learning médico: Explorar estrategias de transferencia de aprendizaje con modelos preentrenados en dominios biomédicos complejos en lugar de inicializar los pesos de la red de forma aleatoria.

## Archivos generados

| Archivo | Descripción |
|---|---|
| `mejor_modelo_bloodmnist.pth` | Pesos del mejor modelo (mayor precisión de validación) guardados durante el entrenamiento. |
| `bloodmnist.npz` | Archivo de datos descargado automáticamente por la librería `medmnist` (train/val/test). |

---
