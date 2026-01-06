import os
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIR = os.path.join(BASE_DIR, 'brain_tumor_dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'detector', 'dl', 'cnn_model.h5')


def train():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Dataset not found at {DATASET_DIR}")
        return

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2], 
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    print(f"📂 Loading data from: {DATASET_DIR}")

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )


    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    base_model.trainable = False

    # 3. Add Custom Layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Better than Flatten() for modern models
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x) # Drop 50% of neurons to prevent memorization
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # 4. Compile
    # We use a lower learning rate (0.0001) because we are fine-tuning
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])

    # 5. Train
    print("🚀 Starting training with Transfer Learning (MobileNetV2)...")
    history = model.fit(
        train_generator,
        epochs=10, # MobileNet learns faster, so 10-15 epochs is usually enough
        validation_data=validation_generator
    )

    # 6. Save
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved successfully at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()