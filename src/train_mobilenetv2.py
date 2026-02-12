import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import tf_keras as keras
from tf_keras.applications import MobileNetV2
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tf_keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "ferplus"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "validation"
    TEST_DIR = DATA_DIR / "test"
    
    MODELS_DIR = BASE_DIR / "models"
    PLOTS_DIR = BASE_DIR / "plots"
    
    IMG_SIZE = 224
    BATCH_SIZE = 128 
    NUM_CLASSES = 6
    
    CLASS_NAMES = ['angry', 'contempt', 'happy', 'neutral', 'sad', 'suprise']
    
    WARMUP_EPOCHS = 15
    WARMUP_LR = 0.001 
    
    FINETUNE_EPOCHS = 35
    FINETUNE_LR = 1e-4 
    
    DROPOUT_RATE = 0.5 
    L2_WEIGHT_DECAY = 1e-5 
    
    ROTATION_RANGE = 20 
    ZOOM_RANGE = 0.2
    
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 7 
    REDUCE_LR_FACTOR = 0.5
    
    MODEL_BEST = "mobilenetv2_ferplus_best.h5"
    MODEL_FINAL = "mobilenetv2_ferplus_final.h5"
    
    def __init__(self):
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def create_data_generators(config):
    print("\n" + "="*60)
    print("CARREGANDO DADOS")
    print("="*60)
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=config.ROTATION_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    print(f"\nTreino: {train_generator.samples} amostras")
    print(f"Validação: {val_generator.samples} amostras")
    print(f"Teste: {test_generator.samples} amostras")
    print(f"Batch: {config.BATCH_SIZE}")
    print(f"Augmentação: Rotação ±{config.ROTATION_RANGE}°, Zoom {config.ZOOM_RANGE}")
    print(f"Classes: {train_generator.class_indices}")
    
    class_weights = compute_class_weights(train_generator)
    print(f"\nPesos calculados:")
    for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        print(f"  {class_name}: {class_weights[class_idx]:.3f}")
    
    return train_generator, val_generator, test_generator, class_weights


def compute_class_weights(generator):
    from sklearn.utils.class_weight import compute_class_weight
    
    class_indices = generator.class_indices
    num_classes = len(class_indices)
    
    class_counts = {}
    for class_name, class_idx in class_indices.items():
        class_dir = generator.directory / class_name
        count = len(list(class_dir.glob('*')))
        class_counts[class_idx] = count
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_idx in range(num_classes):
        class_weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
    
    return class_weights

def build_model(config):
    print("\n" + "="*60)
    print("CONSTRUINDO MODELO")
    print("="*60)
    
    base_model = MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    outputs = Dense(
        config.NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(config.L2_WEIGHT_DECAY),
        name='predictions'
    )(x)
    
    model = keras.Model(inputs, outputs, name='MobileNetV2_FERPlus')
    
    optimizer = Adam(learning_rate=config.WARMUP_LR)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    print(f"\n✓ Modelo construído")
    print(f"✓ Parâmetros: {model.count_params():,}")
    print(f"✓ Treináveis: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model

def train_phase1_warmup(model, train_gen, val_gen, config, class_weights=None):
    print("\n" + "="*60)
    print("FASE 1: AQUECIMENTO")
    print("="*60)
    print(f"Épocas: {config.WARMUP_EPOCHS}")
    print(f"Taxa de aprendizado: {config.WARMUP_LR}")
    print(f"Parâmetros treináveis: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print("="*60 + "\n")
    
    model.compile(
        optimizer=Adam(learning_rate=config.WARMUP_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "warmup_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(
            filename=str(config.PLOTS_DIR / 'warmup_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.WARMUP_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print(f"\n✓ Fase 1 concluída")
    return history


def train_phase2_finetune(model, train_gen, val_gen, config, class_weights=None):
    print("\n" + "="*60)
    print("FASE 2: AJUSTE FINO")
    print("="*60)
    
    base_model = model.layers[1]
    
    base_model.trainable = True
    
    print(f"✓ Backbone descongelado")
    print(f"✓ Camadas: {len(base_model.layers)}")
    print(f"✓ Parâmetros treináveis: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    model.compile(
        optimizer=Adam(learning_rate=config.FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    print(f"\nÉpocas: {config.FINETUNE_EPOCHS}")
    print(f"Taxa de aprendizado: {config.FINETUNE_LR}")
    print("="*60 + "\n")
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(config.MODELS_DIR / config.MODEL_BEST),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            filename=str(config.PLOTS_DIR / 'finetune_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.FINETUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    model.save(str(config.MODELS_DIR / config.MODEL_FINAL))
    print(f"\n✓ Fase 2 concluída - Modelo salvo")
    
    return history

def evaluate_model(model, test_gen, config):
    print("\n" + "="*60)
    print("AVALIANDO MODELO")
    print("="*60)
    
    test_loss, test_acc, test_top2_acc = model.evaluate(test_gen, verbose=1)
    
    print(f"\n✓ Perda: {test_loss:.4f}")
    print(f"✓ Acurácia: {test_acc*100:.2f}%")
    print(f"✓ Top-2: {test_top2_acc*100:.2f}%")
    
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    print("\n" + "="*60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("="*60)
    print(classification_report(
        true_classes,
        predicted_classes,
        target_names=config.CLASS_NAMES,
        digits=4
    ))
    
    return test_loss, test_acc, predicted_classes, true_classes


def plot_training_history(history, config):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ Histórico salvo: training_history.png")
    plt.close()


def plot_confusion_matrix(true_labels, predicted_labels, config):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Matriz de confusão salva: confusion_matrix.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("TREINAMENTO MOBILENETV2")
    print("Dataset FERPlus - 6 Emoções")
    print("="*60)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    
    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        if not split_dir.exists():
            print(f"\n✗ ERRO: Diretório não encontrado: {split_dir}")
            sys.exit(1)
    
    print(f"\ntf_keras: {keras.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    
    train_gen, val_gen, test_gen, class_weights = create_data_generators(config)
    
    model = build_model(config)
    
    history_warmup = train_phase1_warmup(model, train_gen, val_gen, config, class_weights)
    
    history_finetune = train_phase2_finetune(model, train_gen, val_gen, config, class_weights)
    
    combined_history = keras.callbacks.History()
    combined_history.history = {}
    for key in history_warmup.history.keys():
        combined_history.history[key] = (
            history_warmup.history[key] + history_finetune.history[key]
        )
    
    print(f"\nCarregando melhor modelo: {config.MODEL_BEST}")
    best_model = keras.models.load_model(str(config.MODELS_DIR / config.MODEL_BEST))
    
    test_loss, test_acc, pred_labels, true_labels = evaluate_model(best_model, test_gen, config)
    
    plot_training_history(combined_history, config)
    plot_confusion_matrix(true_labels, pred_labels, config)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO")
    print("="*60)
    print(f"Melhor modelo: {config.MODELS_DIR / config.MODEL_BEST}")
    print(f"Modelo final: {config.MODELS_DIR / config.MODEL_FINAL}")
    print(f"Acurácia: {test_acc*100:.2f}%")
    print(f"Gráficos em: {config.PLOTS_DIR}")
    print(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    print("\n" + "="*60)
    print("PRÓXIMO PASSO: CONVERTER PARA TENSORFLOW.JS")
    print("="*60)
    print("Execute:")
    print(f"\ntensorflowjs_converter --input_format=keras \\")
    print(f"    {config.MODELS_DIR / config.MODEL_BEST} \\")
    print(f"    {config.BASE_DIR / 'src/web/model/mobilenet/'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
