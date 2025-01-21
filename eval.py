import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_evaluate_models(dataset_path, model_dir="model_outputs", batch_size=16):
    """
    Load trained models and evaluate them with detailed metrics
    """
    try:
        # Create data generator for evaluation
        datagen = ImageDataGenerator(rescale=1.0/255)
        
        # Load evaluation dataset
        eval_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load models
        model_path = Path(model_dir)
        autoencoder = load_model(str(model_path / 'final_autoencoder.keras'))
        classification_model = load_model(str(model_path / 'final_classification_model.keras'))
        
        logger.info("Models loaded successfully")
        
        # Evaluate classification model
        logger.info("Evaluating classification model...")
        evaluation = classification_model.evaluate(eval_generator)
        logger.info(f"Test Loss: {evaluation[0]:.4f}")
        logger.info(f"Test Accuracy: {evaluation[1]:.4f}")
        
        # Generate predictions
        predictions = classification_model.predict(eval_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = eval_generator.classes[:len(y_pred)]
        
        # Get class labels
        class_labels = list(eval_generator.class_indices.keys())
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(str(model_path / 'confusion_matrix.png'))
        plt.close()
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Plot per-class accuracy
        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_labels, y=per_class_accuracy)
        plt.title('Per-Class Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(str(model_path / 'per_class_accuracy.png'))
        plt.close()
        
        # Print detailed classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))
        
        # Test autoencoder reconstruction
        logger.info("Testing autoencoder reconstruction...")
        sample_batch = next(eval_generator)[0][:5]  # Get 5 sample images
        reconstructed = autoencoder.predict(sample_batch)
        
        # Plot original vs reconstructed images
        plt.figure(figsize=(15, 6))
        for i in range(5):
            # Original
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample_batch[i])
            plt.title('Original')
            plt.axis('off')
            
            # Reconstructed
            plt.subplot(2, 5, i + 6)
            plt.imshow(reconstructed[i])
            plt.title('Reconstructed')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(str(model_path / 'autoencoder_samples.png'))
        plt.close()
        
        # Save metrics to file
        with open(str(model_path / 'evaluation_metrics.txt'), 'w') as f:
            f.write(f"Overall Accuracy: {evaluation[1]:.4f}\n\n")
            f.write("Per-Class Accuracy:\n")
            for label, acc in zip(class_labels, per_class_accuracy):
                f.write(f"{label}: {acc:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_true, y_pred, target_names=class_labels))
        
        return {
            'accuracy': evaluation[1],
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_accuracy,
            'class_labels': class_labels
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Use the same dataset path as in the training script
        dataset_path = r"C:\Users\sanvi\OneDrive\Desktop\Resized_Normalized_Dataset"
        results = load_and_evaluate_models(dataset_path)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        exit(1)