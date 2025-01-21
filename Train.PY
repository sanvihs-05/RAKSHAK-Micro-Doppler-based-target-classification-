import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Dense, Flatten, Input, Conv2D, MaxPooling2D, 
    UpSampling2D, BatchNormalization, Dropout
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metric 
import confusion_matrix, classification_report
import numpy as np
import os
import logging
import sys
from pathlib import Path
import tensorflow.keras.backend as K

tf.get_logger().setLevel('ERROR')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_autoencoder(input_shape=(224, 224, 3)):
    """Create autoencoder with simplified architecture"""
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(input_img, decoded)

def create_classification_model(num_classes):
    """Create classification model"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers:
        layer.trainable = False
    
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=output)
def limit_memory_growth():
    """Configure GPU memory growth to prevent OOM errors"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        else:
            logger.info("No GPUs found. Running on CPU.")
    except Exception as e:
        logger.warning(f"Error configuring GPU: {str(e)}")
class DenoisingDataGenerator:
    def __init__(self, image_generator, noise_factor=0.3, batch_size=32):
        self.image_generator = image_generator
        self.noise_factor = noise_factor
        self.batch_size = batch_size
        self.steps = self.image_generator.samples // self.batch_size

    def flow(self):
        while True:
            try:
                batch = next(self.image_generator)[0]
                if batch.shape[0] == 0:
                    continue
                    
                noisy_batch = batch + self.noise_factor * np.random.normal(
                    loc=0.0, scale=1.0, size=batch.shape
                )
                noisy_batch = np.clip(noisy_batch, 0., 1.)
                
                if np.any(np.isnan(noisy_batch)) or np.any(np.isnan(batch)):
                    continue
                    
                yield noisy_batch, batch
            except Exception as e:
                logger.error(f"Error in data generator: {str(e)}")
                continue

def check_dataset(dataset_path):
    """Verify dataset structure and contents"""
    path = Path(dataset_path)
    if not path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    classes = [d for d in path.iterdir() if d.is_dir()]
    if not classes:
        raise ValueError(f"No class directories found in {dataset_path}")
    
    total_images = 0
    for class_dir in classes:
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        if not images:
            raise ValueError(f"No images found in class directory: {class_dir}")
        total_images += len(images)
    
    logger.info(f"Found {len(classes)} classes with total {total_images} images")
    return len(classes)

def create_data_generators(dataset_path, batch_size=32):
    """Create data generators with validation"""
    try:
        datagen = ImageDataGenerator(
            rescale=1.0/255,
            validation_split=0.2,
            preprocessing_function=lambda x: np.clip(x, 0, 255)
        )
        
        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    except Exception as e:
        logger.error(f"Error creating data generators: {str(e)}")
        raise

def main():
    try:
        # Configure GPU memory growth
        limit_memory_growth()
        
        # Create output directory if it doesn't exist
        output_dir = Path("model_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Dataset path
        dataset_path = r"C:\Users\sanvi\OneDrive\Desktop\Resized_Normalized_Dataset"
        
        # Check dataset
        num_classes = check_dataset(dataset_path)
        
        # Create data generators with smaller batch size
        batch_size = 16  # Reduced batch size
        train_generator, validation_generator = create_data_generators(dataset_path, batch_size)
        
        # Create models
        autoencoder = create_autoencoder()
        classification_model = create_classification_model(num_classes)
        
        # Compile models
        autoencoder.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse',
            metrics=['mae']
        )
        
        classification_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate steps
        steps_per_epoch = train_generator.samples // batch_size
        validation_steps = validation_generator.samples // batch_size
        
        # Training callbacks
        autoencoder_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'autoencoder_checkpoint.keras'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        classification_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(output_dir / 'classification_checkpoint.keras'),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train autoencoder
        logger.info("Training Autoencoder...")
        train_denoising_gen = DenoisingDataGenerator(train_generator, batch_size=batch_size)
        val_denoising_gen = DenoisingDataGenerator(validation_generator, batch_size=batch_size)
        
        autoencoder_history = autoencoder.fit(
            train_denoising_gen.flow(),
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=val_denoising_gen.flow(),
            validation_steps=validation_steps,
            callbacks=autoencoder_callbacks,
            verbose=1
        )
        
        # Save final autoencoder
        autoencoder.save(str(output_dir / 'final_autoencoder.keras'))
        
        # Train classification model
        logger.info("Training Classification Model...")
        classification_history = classification_model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=classification_callbacks,
            verbose=1
        )
        
        # Save final classification model
        classification_model.save(str(output_dir / 'final_classification_model.keras'))
        
        # Evaluation
        logger.info("Evaluating Model...")
        test_loss, test_acc = classification_model.evaluate(
            validation_generator,
            steps=validation_steps
        )
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        # Generate predictions
        predictions = classification_model.predict(
            validation_generator,
            steps=validation_steps
        )
        y_pred = np.argmax(predictions, axis=1)
        y_true = validation_generator.classes[:len(y_pred)]
        
        # Print classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        return {
            'autoencoder': autoencoder,
            'classification_model': classification_model,
            'autoencoder_history': autoencoder_history,
            'classification_history': classification_history
        }
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Clear session
        K.clear_session()

if __name__ == "__main__":
    try:
        results = main()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        sys.exit(1)  