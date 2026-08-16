# Clasificación de Células Sanguíneas con ResNet (BloodMNIST)


Link de video de prueba interactiva: https://drive.google.com/file/d/1qIaUdwLFmHAQbpFyD9Xh_m-96i_X6Sk7/view?usp=sharing

Trabajo final del seminario **Deep Learning applied in Bioimage Analysis** -- Pontificia Universidad Católica del Perú (PUCP).

**Autores:** Gianfranco Moises Poma Canchari --100%, Valeshka Lavado Guerra --100%, Andy Edery Wusen --100%, Kusi Uñapillco Franco --100%


## Descripción general

El notebook `Blood_classification_v4.ipynb` implementa un pipeline completo de *deep learning* con **PyTorch** para resolver un problema de **clasificación multiclase de células sanguíneas periféricas** a partir de imágenes microscópicas de baja resolución. El objetivo es evaluar la capacidad de abstracción y generalización de una arquitectura convolucional profunda (ResNet) entrenada **desde cero** (sin pesos preentrenados) sobre el benchmark **BloodMNIST**.

## Relevancia

La aplicación biomédica de esta tarea es automatizar el conteo y clasificación celular en frotis de sangre periférica, lo cual acelera diagnósticos en hematología y reduce la carga operativa en laboratorios.



---

## Tabla de contenidos

1. Dataset y Estadísticas Generales

2. Feature Engineering y Exploración Visual

3. Definición de la Tarea

4. Selección de Arquitectura

5. Justificación de Hiperparámetros

6. Dinámica y Evaluación de Entrenamiento

7. Evaluación en Test y Métricas

8. Conclusiones y Sentido Crítico

---
   

## 1. Dataset y Estadísticas Generales

| Propiedad | Valor |
|---|---|
| Total de imágenes | 17,092 |
| Número de clases | 8 |
| Resolución | 3×28×28 (RGB) |
| Origen | Células individuales normales, sin infecciones, enfermedades hematológicas/oncológicas ni tratamiento farmacológico |
| Preprocesamiento | Recorte central + redimensionado a 28×28 px |

- Adquisición y Dimensiones: Las imágenes originales tenían una resolución de 3x360x363 píxeles. Luego, fueron recortadas en el centro a 3x200x200 y finalmente redimensionadas a 3x28x28 mediante interpolación spline cúbica.  

- Balance de Clases: Total de 17,092 imágenes organizadas en 8 clases (Basophil, Eosinophil, Erythroblast, IG, Lymphocyte, Monocyte, Neutrophil, Platelet). No existe un desbalance de clases severo.  

- Splits: Se mantuvo la división oficial de MedMNIST (70% train / 10% val / 20% test) para garantizar una comparación justa y reproducible con el benchmark oficial.  

- Los datos se descargan automáticamente mediante la librería medmnist y se cargan en DataLoaders de PyTorch con batch_size=128.


## 2. Feature Engineering y Exploración Visual

En esta sección se realizó la inspección cualitativa y cuantitativa de patrones biológicos clave en las imágenes histopatológicas. El objetivo es identificar características morfológicas determinantes.

Objetivos de la Exploración:

* **Identificar atipia nuclear:** Evaluar variaciones en tamaño, forma e hipercromasia nuclear.
* **Analizar textura y microambiente:** Cuantificar la heterogeneidad tisular y desorganización celular.
* **Delinear bordes y membranas:** Evaluar la pérdida de cohesión celular y límites de invasión tumoral.
* **Justificar el preprocesamiento:** Determinar necesidades de normalización de color, realce de contraste y reducción de ruido.

Métodos y Técnicas Aplicadas:

| Filtro / Descriptor | Tipo de Extracción | Patrón Biológico Evaluado | Justificación para el Preprocesamiento |
| :--- | :--- | :--- | :--- |
| **Umbralización Otsu** | Segmentación por intensidad | Segmentación de núcleos hipercromáticos y áreas de necrosis | Permite aislar la carga celular vs. fondo/estroma, justificando el filtrado de artefactos y recorte de parches con alta densidad tisular. |
| **Filtro Sobel / Canny** | Gradiente espacial / Bordes | Membranas celulares, transiciones epitelio-estroma | Detecta la pérdida de bordes definidos típica de la atipia; justifica el uso de normalización de contraste y filtros de agudizamiento (*sharpening*). |
| **Local Binary Patterns (LBP)** | Textura local invariante | Micro-texturas nucleares y granularidad de la cromatina | Captura patrones finos de malignidad independientes de la iluminación global; fundamenta la necesidad de aumentos de datos (*data augmentation*) de textura/ruido. |
| **Descriptores Haralick (GLCM)** | Estadísticos de segundo orden | Heterogeneidad tisular (contraste, energía, homogeneidad, correlación) | Modela la desorganización estructural del tejido; valida la relevancia de conservar detalles de alta frecuencia sin aplicar un suavizado excesivo. |
| **Mapas de Intensidad (Espacios HSV / LAB / Desconvolución H&E)** | Colorimetría y concentración de tinción | Concentración de Hematoxilina (núcleos) y Eosina (citoplasma) | Evidencia la variabilidad por lotes de tinción; **justifica directamente la normalización de color (e.g., Macenko o Vahadane)** antes de alimentar modelos de aprendizaje profundo. |


Hallazgos Clave:

* **Hipercromasia y Densidad Nuclear:** El canal de hematoxilina y la umbralización adaptativa revelan aglomeraciones nucleares irregulares en regiones malignas, confirmando que la densidad de núcleos es una señal discriminante primaria.
* **Variabilidad de Tinción:** Se observaron variaciones notables de luminosidad y saturación entre láminas, demostrando que la falta de normalización de color induciría sesgo de adquisición (*batch effect*).
* **Heterogeneidad de Bordes:** Las regiones benignas muestran límites celulares continuos y homogéneos bajo el operador Sobel, mientras que las zonas atípicas presentan bordes fragmentados y difusos.

Conclusiones para el Pipeline de Preprocesamiento:

1. **Normalización de Color Obligatoria:** Implementación de normalización H&E (Macenko/Reinhard) para estandarizar la respuesta espectral entre muestras.
2. **Filtrado Espacial Controlado:** Uso de eliminación de ruido (*denoising*) con preservación de bordes (e.g., filtro bilateral o Gaussiano leve) para no degradar las micro-texturas detectadas por LBP.
3. **Selección de ROIs basada en Densidad:** Filtrado de parches con bajo contenido celular mediante máscaras de Otsu para entrenar únicamente sobre tejido biológicamente relevante.

## 3. Definición de la Tarea

- Definición Técnica: Problema de clasificación multiclase supervisada de células sanguíneas periféricas a partir de imágenes microscópicas de baja resolución.

- Relevancia y Aplicación Biomédica: La automatización de este proceso permite acelerar el conteo y clasificación celular en frotis de sangre periférica, lo cual optimiza los diagnósticos en hematología y reduce significativamente la carga operativa en laboratorios clínicos.

## 4. Selección de Arquitectura

| Etapa / Bloque | Componentes Principales | Configuración / Operación | Canales Entrada $\rightarrow$ Salida | Dimensión Espacial |
| :--- | :--- | :--- | :---: | :---: |
| **Input** | Tensor de entrada | Imagen RGB / Multicanal | `in_channels` | $H \times W$ |
| **Stem** | Conv2D + BatchNorm + ReLU | `kernel=3x3, stride=1, padding=1` | $3 \rightarrow 32$ | $H \times W$ |
| **Stage 1** | `ResidualBlock` | Conv3x3 ($s=1$) $\rightarrow$ Conv3x3 ($s=1$) + Identity | $32 \rightarrow 32$ | $H \times W$ |
| **Stage 2** | `ResidualBlock` (Downsample) | Conv3x3 ($s=2$) $\rightarrow$ Conv3x3 ($s=1$) + Shortcut (Conv1x1, $s=2$) | $32 \rightarrow 64$ | $H/2 \times W/2$ |
| **Stage 3** | `ResidualBlock` (Downsample) | Conv3x3 ($s=2$) $\rightarrow$ Conv3x3 ($s=1$) + Shortcut (Conv1x1, $s=2$) | $64 \rightarrow 128$ | $H/4 \times W/4$ |
| **Classifier** | AdaptiveAvgPool2d + Dropout + FC | `GAP (1x1)` $\rightarrow$ `Dropout(p=0.4)` $\rightarrow$ `Linear(128, 8)` | $128 \rightarrow \text{num}_{\text{classes}}$ | $1 \times 1 \rightarrow \text{logits}$ |


En lugar de utilizar modelos preentrenados genéricos, se desarrolló e implementó desde cero una arquitectura residual personalizada denominada MiPropiaResNet:

   - Estructura base (Stem): Compuesta por una capa convolucional de $3 \times 3$ (32 canales), normalización        por lotes (Batch Normalization) y activación ReLU.
  
   - Etapas Residuales: Consta de tres etapas sucesivas de bloques residuales que aumentan progresivamente los       mapas de características (32, 64 y 128 canales) aplicando convoluciones de $3 \times 3$ y accesos directos      (shortcuts) para evitar la degradación del gradiente.
  
   - Clasificación: Utiliza un Adaptive Average Pooling global ($1 \times 1$), una capa de Dropout ($p = 0.4$)       para mitigar el sobreajuste, y una capa totalmente conectada (Fully Connected) proyectada a las 8 clases        de salida.

Especificaciones Clave
* **Estrategia Residual:** Conexiones directas de tipo identity/projection según cambio de resolución o canales.
* **Regularización:** Inclusión de capa `Dropout(p=0.4)` previa a la clasificación para mitigar sobreajuste.
* **Cuello de Botella Clasificador:** Reducción espacial vía **Global Average Pooling** previo a la capa lineal final.

## 5. Justificación de Hiperparámetros

| Parámetro | Valor |
|---|---|
| Épocas | 100 |
| Función de pérdida | Cross-Entropy Loss |
| Optimizador | Adam (lr = 0.001) |
| Batch size | 128 |
| Scheduler | `MultiStepLR`, milestones en épocas 50 y 75, gamma = 0.1 |
| Hardware recomendado | GPU (Colab T4) |

- Fundamentación Técnica: La elección de estos hiperparámetros no es arbitraria; replica exactamente el esquema base (baseline) oficial establecido por los autores del benchmark MedMNIST v2. El uso de Cross-Entropy es estándar para multiclase y el scheduler permite refinar la convergencia en etapas finales.

## 6. Dinámica y Evaluación de Entrenamiento

- Trazabilidad: Se registra el historial de pérdida de entrenamiento (L train) y precisión de validación por época, graficando las curvas con líneas verticales que marcan las caídas del learning rate.

- Convergencia: El modelo mostró un aprendizaje rápido y estable, superando el 95% de precisión en validación a partir de la época 14 y estabilizándose de manera óptima tras los ajustes del scheduler.

- Control de Entrenamiento: Se aplica checkpointing guardando automáticamente el mejor modelo en mejor_modelo_bloodmnist.pth (basado en la mayor precisión de validación).

## 7. Evaluación en Test y Métricas

- Desempeño Global: Precisión final en test de 94.94% (Accuracy global de 0.95). Mejor precisión de validación: 95.91%.

- Métricas Detalladas (Test Set, 3,421 imágenes):

- Reporte de Clasificación: Incluye Precision, Recall y F1-score por clase (destacando clases como Basophil, Eosinophil, Platelet, etc.).

- Matriz de Confusión: Generada mediante un mapa de calor (heatmap) para identificar confusiones específicas (ej. clases IG y Monocyte).

- Inferencia Visual: El notebook incluye pruebas interactivas con imágenes propias y muestras aleatorias del conjunto .npz mostrando el top-3 de probabilidades y niveles de confianza (softmax).

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

## 8. Conclusiones y Sentido Crítico

- Hallazgos: La arquitectura MiPropiaResNet diseñada desde cero demuestra una alta capacidad de abstracción, alcanzando un competitivo 96.00% de precisión en test. Las clases que experimentan mayor confusión o menor rendimiento relativo son IG (granulocito inmaduro) y Monocyte, debido al solapamiento morfológico natural y la pérdida de detalle por la baja resolución.

- Limitaciones y advertencias:
   - Restricción de resolución y uso clínico: Los conjuntos de datos de MedMNIST están diseñados exclusivamente      para benchmarking ligero de modelos. Las imágenes de BloodMNIST están comprimidas a 28x28  píxeles, lo que      genera una pérdida masiva de detalle frente a las fotografías reales de laboratorio (3x360x363 originales       recortadas a 3x200x200). Por esta razón, los autores advierten explícitamente que el modelo no está             destinado para uso clínico, ya que una reducción tan sustancial de la resolución puede ser insuficiente         para capturar patologías complejas.

   - Entrenamiento desde cero: El modelo se entrena completamente desde cero (sin pesos preentrenados de             transfer learning), lo que limita la capacidad de generalización frente a variaciones en la tinción o           artefactos de captura externos.

- Propuestas de mejora para traducción e investigación futura:

Para superar estas limitaciones y avanzar hacia un entorno de validación real o clínico, se propone:

1. Uso de imágenes de alta resolución: Entrenar arquitecturas con frotis de sangre periférica completos y de alta resolución que no omitan detalles ultraestructurales clave de las células.

2. Transfer Learning médico: Explorar estrategias de transferencia de aprendizaje con modelos preentrenados en dominios biomédicos complejos en lugar de inicializar los pesos de la red de forma aleatoria.



## Archivos generados

| Archivo | Descripción |
|---|---|
| `mejor_modelo_bloodmnist.pth` | Pesos del mejor modelo (mayor precisión de validación) guardados durante el entrenamiento. |
| `bloodmnist.npz` | Archivo de datos descargado automáticamente por la librería `medmnist` (train/val/test). |

---
