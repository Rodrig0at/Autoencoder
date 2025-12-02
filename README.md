# Autoencoder U-Net para Reconstrucción de Imágenes

Este repositorio contiene la implementación de un autoencoder basado en arquitectura U-Net con conexiones residuales (skip connections) para la reconstrucción de imágenes RGB de 224×224 píxeles. El proyecto incluye tanto un modelo clásico básico como una versión avanzada U-Net que incorpora una función de pérdida híbrida combinando MSE y SSIM.

## 📋 Contenido del Repositorio
- **`Papper_Autoencoder.pdf`**: Explicación a detalle sobre los modelos realizados.
- **`EmbeddingConv.ipynb`**: Notebook principal que contiene la implementación completa del autoencoder U-Net, incluyendo el entrenamiento, evaluación y visualización de resultados.
- **`Embeddingsautoencoders.ipynb`**: Notebook con la implementación del autoencoder clásico (modelo base para comparación).
- **`Queries.py`**: Script de línea de comandos para realizar inferencia con los modelos entrenados, soportando tanto el modelo clásico como el U-Net.

## 🏗️ Arquitectura del Modelo

### Autoencoder U-Net

El modelo U-Net implementado utiliza una arquitectura simétrica encoder-decoder con cinco niveles de profundidad, caracterizada por:

**Encoder:**

- Cinco bloques convolucionales con reducción progresiva de resolución (224→112→56→28→14 píxeles)
- Filtros que incrementan en profundidad: 32, 64, 128, 256, 512 canales
- Normalización por lotes (Batch Normalization) en cada bloque
- Dropout incremental (0.1, 0.1, 0.2, 0.2) para regularización
- Bottleneck central de 14×14×512

**Decoder:**

- Arquitectura simétrica con upsampling progresivo (14→28→56→112→224 píxeles)
- Concatenación de skip connections desde el encoder para preservar detalles espaciales
- Reducción gradual de canales: 512→256→128→64→32
- Dropout matching al encoder
- Capa final con activación sigmoid para generar salida RGB normalizada

**Parámetros totales:** ~12.5 millones

### Función de Pérdida Combinada

El modelo utiliza una función de pérdida híbrida que balancea la precisión pixel a pixel con la calidad perceptual:

```python
Loss = 0.7 × MSE + 0.3 × (1 - SSIM)
```

Esta combinación permite que el modelo capture tanto la fidelidad numérica (MSE) como la similitud estructural (SSIM), reduciendo el efecto de borrosidad típico de las pérdidas basadas únicamente en MSE y mejorando la percepción visual de las reconstrucciones.

## 📊 Resultados del Entrenamiento

El modelo fue entrenado durante 50 épocas con las siguientes configuraciones y resultados:

**Configuración:**

- Optimizador: Adam (learning rate = 0.0001)
- Batch size: 32
- Split de validación: 30%
- Early Stopping: patience=15, min_delta=0.0001
- ReduceLROnPlateau: factor=0.5, patience=5

**Métricas Finales:**

- Pérdida de entrenamiento: 0.07
- Pérdida de validación: 0.08
- MAE de entrenamiento: 0.06
- MAE de validación: 0.11

**Interpretación de Resultados:**

Las gráficas de entrenamiento muestran que tanto la curva de pérdida de entrenamiento como la de validación descienden de manera pronunciada desde aproximadamente 0.36 hasta estabilizarse alrededor de 0.08, evidenciando una convergencia similar entre ambas que indica un aprendizaje efectivo sin signos severos de sobreajuste. Sin embargo, el Error Absoluto Medio presenta una brecha moderada entre entrenamiento (0.09) y validación (0.12), sugiriendo la presencia de un ligero overfitting, aunque el modelo logra generalizar razonablemente bien en datos no vistos durante el entrenamiento. Esta discrepancia es común en modelos de alta capacidad y puede mitigarse mediante incremento en dropout, aumento de datos, regularización L2 adicional o ajuste fino de los callbacks de early stopping.

## 🚀 Instalación y Requisitos

### Dependencias Principales

```bash
pip install tensorflow==2.10.0
pip install tensorflow-probability
pip install numpy==1.26.4
pip install matplotlib
pip install seaborn
pip install pillow
```

### Configuración GPU (Opcional - Windows)

Para habilitar aceleración GPU en Windows:

1. Instalar [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
2. Instalar [cuDNN 8.1](https://developer.nvidia.com/cudnn)
3. Reinstalar NumPy compatible:
   ```bash
   pip install numpy==1.26.4 --force-reinstall
   ```

## 📁 Estructura de Datos

El modelo espera que el dataset esté organizado de la siguiente manera:

```
images/
├── clase1/
│   ├── imagen1.jpg
│   ├── imagen2.jpg
│   └── ...
├── clase2/
│   ├── imagen1.jpg
│   └── ...
└── ...
```

**Nota:** Aunque el autoencoder no utiliza las etiquetas de clase, ImageDataGenerator requiere esta estructura de directorios.

## 💻 Uso

### Entrenamiento

1. Abrir `EmbeddingConv.ipynb` en Jupyter Notebook o Google Colab
2. Actualizar la ruta del dataset en la celda correspondiente:
   ```python
   dataset_path = 'ruta/a/tu/dataset/images'
   ```
3. Ejecutar todas las celdas secuencialmente

El notebook guardará automáticamente los modelos entrenados en formato `.h5`:

- `autoencoder_unet.h5` - Modelo completo
- `encoder_unet_.h5` - Solo encoder
- `decoder_unet_.h5` - Solo decoder
- `*_weights.h5` - Versiones solo con pesos

### Inferencia con Script

El script `Queries.py` permite realizar inferencia desde la línea de comandos:

```bash
# Reconstruir imagen con modelo U-Net
python Queries.py --ruta imagen.jpg --modelo unet --modo autoencoder --mostrar

# Codificar imagen (obtener representación latente)
python Queries.py --ruta imagen.jpg --modelo unet --modo encoder

# Decodificar desde espacio latente
python Queries.py --ruta imagen.jpg --modelo unet --modo decoder

# Usar modelo clásico
python Queries.py --ruta imagen.jpg --modelo clasico --modo autoencoder --mostrar
```

**Parámetros:**

- `--ruta`: Ruta a la imagen de entrada (requerido)
- `--modelo`: Tipo de modelo (`clasico` o `unet`, default: `unet`)
- `--modo`: Operación a realizar (`autoencoder`, `encoder`, `decoder`, default: `autoencoder`)
- `--mostrar`: Mostrar visualización de imagen original vs reconstruida (solo modo autoencoder)

## 📈 Evaluación del Modelo

El notebook incluye evaluación completa con las siguientes métricas por imagen:

- **MSE (Mean Squared Error):** Error cuadrático medio pixel a pixel
- **MAE (Mean Absolute Error):** Error absoluto medio
- **SSIM (Structural Similarity Index):** Similitud estructural (0-1, valores más altos = mejor)

Ejemplo de salida:

```
MÉTRICAS DE RECONSTRUCCIÓN U-NET

Imagen 1:
  MSE:  0.003421
  MAE:  0.042156
  SSIM: 0.9234

PROMEDIOS:
  MSE promedio:  0.003856
  MAE promedio:  0.045321
  SSIM promedio: 0.9156
```

## 🔍 Comparación: Modelo Clásico vs U-Net

| Característica         | Modelo Clásico    | U-Net         |
| ---------------------- | ----------------- | ------------- |
| Resolución entrada     | 28×28 (grayscale) | 224×224 (RGB) |
| Skip connections       | ❌ No             | ✅ Sí         |
| Función de pérdida     | MSE               | MSE + SSIM    |
| Parámetros             | ~2M               | ~12.5M        |
| Calidad reconstrucción | Moderada          | Alta          |
| Preservación detalles  | Baja              | Alta          |
| Tiempo inferencia      | Rápido            | Moderado      |

## 🛠️ Mejoras Futuras

Basado en el análisis de overfitting detectado, se recomiendan las siguientes mejoras:

1. **Regularización adicional:**
   - Incrementar dropout de 0.1/0.2 a 0.2/0.3
   - Añadir regularización L2 en capas convolucionales
2. **Aumento de datos (Data Augmentation):**
   - Rotaciones aleatorias
   - Flips horizontales/verticales
   - Ajustes de brillo y contraste
3. **Arquitectura:**
   - Experimentar con attention mechanisms
   - Probar variantes como ResNet-Autoencoder
4. **Optimización:**
   - Ajuste fino de hiperparámetros mediante grid search
   - Experimentar con diferentes ratios MSE/SSIM

## 📚 Referencias

- Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
- Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP 2004)
- Documentación TensorFlow: https://www.tensorflow.org/

## 📝 Licencia

Este proyecto está disponible para uso educativo y de investigación.

## 👤 Autor

Rodrigo Antonio Benítez De La Portilla - Proyecto desarrollado como parte de estudios en Deep Learning y Visión por Computadora.

---

**Nota:** Para cualquier pregunta o problema, por favor revisar los notebooks incluidos que contienen comentarios detallados sobre cada paso del proceso.

