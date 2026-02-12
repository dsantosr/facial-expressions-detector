#!/usr/bin/env python3

import tensorflow as tf
import tf_keras as keras
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_H5 = BASE_DIR / "models" / "mobilenetv2_ferplus_best.h5"
SAVEDMODEL_DIR = BASE_DIR / "models" / "mobilenetv2_savedmodel"

print("=" * 60)
print("CONVERTENDO PARA SAVEDMODEL")
print("=" * 60)

print(f"\n✓ Carregando: {MODEL_H5}")
model = keras.models.load_model(str(MODEL_H5))

print(f"✓ Modelo carregado")
print(f"  - Entrada: {model.input_shape}")
print(f"  - Saída: {model.output_shape}")
print(f"  - Parâmetros: {model.count_params():,}")

print(f"\n✓ Salvando em: {SAVEDMODEL_DIR}")
tf.saved_model.save(model, str(SAVEDMODEL_DIR))

print(f"\n✓ SavedModel criado!")
print(f"  - Local: {SAVEDMODEL_DIR}")

print("\n" + "=" * 60)
print("PRÓXIMO PASSO: CONVERTER PARA TENSORFLOW.JS")
print("=" * 60)
print("\nExecute:")
print("\ntensorflowjs_converter \\")
print(f"  --input_format=tf_saved_model \\")
print(f"  {SAVEDMODEL_DIR} \\")
print(f"  {BASE_DIR / 'src/web/model/mobilenet/'}")
print("\n" + "=" * 60)