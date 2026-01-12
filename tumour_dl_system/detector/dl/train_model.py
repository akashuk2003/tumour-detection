import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = os.path.join(BASE_DIR, 'brain_tumor_dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'detector', 'dl', 'cnn_model_new.h5')


def train():
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset not found at {DATASET_DIR}")
        return

    # Data augmentation for training (with rescaling to [0, 1])
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3], 
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Validation data should only be rescaled, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print(f"[INFO] Loading data from: {DATASET_DIR}")

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=16,  # Smaller batch for more updates
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Print class distribution
    print(f"\n[INFO] Class indices: {train_generator.class_indices}")
    print(f"[INFO] Training samples: {train_generator.samples}")
    print(f"[INFO] Validation samples: {validation_generator.samples}")

    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[INFO] Class weights: {class_weight_dict}")

    # 1. Load Pre-trained MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 2. Freeze base model completely for initial training
    base_model.trainable = False
    
    print(f"[INFO] Base model frozen for initial training")

    # 3. Add Custom Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # 4. Compile with higher learning rate for training only top layers
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # 5. Train top layers first
    print("\n[PHASE 1] Training classification head only...")
    print("=" * 60)
    
    model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        class_weight=class_weight_dict
    )

    # 6. Now fine-tune the last 30 layers
    print("\n[PHASE 2] Fine-tuning last 30 layers of MobileNetV2...")
    print("=" * 60)
    
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(learning_rate=0.00001),  # Very low LR for fine-tuning
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator,
        class_weight=class_weight_dict
    )

    # 7. Save model
    model.save(MODEL_SAVE_PATH, save_format='h5')
    
    # 8. Print final metrics
    print("\n" + "=" * 60)
    print("[DONE] Training Complete!")
    print(f"   Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"[SAVED] Model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()