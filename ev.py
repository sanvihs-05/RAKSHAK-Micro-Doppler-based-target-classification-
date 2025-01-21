import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import logging
from pathlib import Path
import pandas as pd

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
        
        # Evaluate classification model
        evaluation = classification_model.evaluate(eval_generator)
        
        # Generate predictions
        predictions = classification_model.predict(eval_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = eval_generator.classes[:len(y_pred)]
        
        # Get class labels
        class_labels = list(eval_generator.class_indices.keys())
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        return {
            'accuracy': evaluation[1],
            'confusion_matrix': cm,
            'class_labels': class_labels,
            'per_class_accuracy': per_class_accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")
    
    st.title("Model Evaluation Dashboard")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    dataset_path = st.sidebar.text_input(
        "Dataset Path", 
        value=r"C:\Users\sanvi\OneDrive\Desktop\Resized_Normalized_Dataset"
    )
    
    if st.sidebar.button("Run Evaluation"):
        try:
            with st.spinner("Evaluating model..."):
                results = load_and_evaluate_models(dataset_path)
                
                # Display overall accuracy
                st.header("Overall Performance")
                accuracy = results['accuracy'] * 100
                st.metric("Model Accuracy", f"{accuracy:.2f}%")
                
                # Create two columns for visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        results['confusion_matrix'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=results['class_labels'],
                        yticklabels=results['class_labels']
                    )
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.subheader("Per-Class Accuracy")
                    per_class_df = pd.DataFrame({
                        'Class': results['class_labels'],
                        'Accuracy': results['per_class_accuracy'] * 100
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=per_class_df, x='Class', y='Accuracy')
                    plt.title('Per-Class Accuracy')
                    plt.ylabel('Accuracy (%)')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                
                # Display classification report
                st.subheader("Detailed Classification Report")
                report = classification_report(
                    results['y_true'], 
                    results['y_pred'], 
                    target_names=results['class_labels']
                )
                st.text(report)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()