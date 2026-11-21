# HemoLearn AI — clasificador de leucocitos

Modelo de visión para la app educativa **HemoLearn AI** (flashcards, quizzes y modo laboratorio).
Clasifica imágenes de **una sola célula** en 5 clases: `basophil`, `eosinophil`, `lymphocyte`, `monocyte` y `neutrophil`.
El modelo correrá dentro del teléfono (Flutter o Kotlin, por definir), así que debe ser ligero.

> ⚠️ Herramienta educativa, **no de diagnóstico**.

## Estado actual

| Paso | Estado |
|---|---|
| Inspección de datos y mapeo de clases | ✅ Hecho |
| Notebook completo (`entrenamiento.ipynb`) | ✅ Hecho |
| Prueba rápida de principio a fin (`MODO_RAPIDO = True`) | ✅ Corre sin errores (≈4 min) |
| **Entrenamiento completo (`MODO_RAPIDO = False`)** | ⏳ **Pendiente** (≈2 h estimadas con una RTX 3050 de 4 GB) |
| Revisar resultados, umbral y modelo final | ⏳ Pendiente |
| Conversión a TFLite | ⏸️ Opcional: no hay TensorFlow en el entorno; ONNX es el formato principal |
| Integración en la app | ⏳ Pendiente (Flutter o Kotlin) |

Las cifras de `outputs/prueba_rapida/` **no son resultados válidos** (1 época, 70 imágenes por clase): solo
demuestran que el flujo funciona. Los resultados reales aparecerán en `outputs/` tras el entrenamiento completo.

## Contenido del repositorio

```
entrenamiento.ipynb        notebook principal (12 secciones)
requirements.txt           versiones exactas del entorno
outputs/prueba_rapida/     salidas de la prueba rápida (figuras, métricas, particiones)
data/                      NO incluido: hay que descargarlo (ver abajo)
```

Secciones del notebook: 1. Contexto y estado del arte · 2. Configuración · 3. Datos · 4. Particiones 70/15/15 ·
5. Preprocesamiento · 6. Entrenamiento por etapas · 7. Evaluación (PBC y Raabin) · 8. Umbral de confianza ·
9. Grad-CAM · 10. Regla de salud · 11. Exportación ONNX · 12. Entregables y conclusiones.

## Cómo reproducirlo

1. **Datos** (no están en el repo por tamaño). Colócalos así:
   ```
   data/PBC_dataset_normal_DIB/<basophil, eosinophil, lymphocyte, monocyte, neutrophil, ...>/
   data/Raabin/TestA/<Basophil, Eosinophil, Lymphocyte, Monocyte, Neutrophil>/
   data/Raabin/Test-B/<Lymphocyte, Neutrophil>/
   ```
   - PBC: Acevedo et al., Mendeley Data, DOI [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1)
   - Raabin-WBC: subconjuntos TestA y Test-B
2. **Entorno** (Python 3.11):
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
   ```
3. Abrir `entrenamiento.ipynb` y ejecutar todo. Para el entrenamiento real, poner `MODO_RAPIDO = False`
   en la primera celda de código; los resultados se guardan en `outputs/`.

## Decisiones tomadas

- **Datos.** Entrenamiento con PBC (5 de sus 8 clases; se ignoran `erythroblast`, `ig` y `platelet`).
  Validación externa con Raabin: **TestA es la principal** (tiene las 5 clases) y **Test-B la secundaria**
  (solo tiene linfocitos y neutrófilos, así que ahí solo se miden esas dos clases).
  Raabin nunca se usa para entrenar, ajustar ni elegir umbrales.
- **Modelos.** MobileNetV2 (pensado para la app), ResNet18 (línea base) y DeiT-Tiny (ViT pequeño). Se usa DeiT-Tiny
  y no `vit_tiny` porque comparte la normalización de ImageNet que usará la app.
- **Entrenamiento.** Dos etapas (cabeza congelada → últimos bloques), pérdida con pesos por clase, precisión mixta,
  mejor modelo según F1 macro en validación.
- **Elección del modelo final** (solo con validación de PBC): el más ligero cuyo F1 macro esté a ≤1 punto del mejor
  (`TOLERANCIA_F1`, ajustable).
- **Preprocesado idéntico al de la app:** 224×224 bilineal sin recorte, RGB, /255, media y desviación de ImageNet,
  NCHW float32. Se guarda en `preprocessing.json` junto con la temperatura de calibración y el umbral.

## Problemas encontrados y resueltos

- En `PBC/neutrophil` hay un archivo oculto de macOS (`.DS_169665.jpg`) que no es una imagen: se ignora.
- `cudnn.benchmark = True` hacía muy lenta la ejecución (≈30 s por cada tamaño de batch nuevo): se desactivó.
- timm inicializa la cabeza de MobileNetV2 con pesos grandes y no aprendía: se reinicializa con pesos pequeños.

## Próximos pasos

1. Ejecutar el entrenamiento completo (`MODO_RAPIDO = False`).
2. Revisar la tabla comparativa, la caída de PBC a Raabin (sobre todo en monocitos) y el umbral propuesto.
3. Decidir Flutter o Kotlin e integrar `outputs/model.onnx` con ONNX Runtime.
