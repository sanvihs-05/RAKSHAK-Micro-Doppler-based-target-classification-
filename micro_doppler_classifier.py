import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define categories
CATEGORIES = [
    '3_long_blades_rotor',
    '3_short_blade_rotor_1',
    '3_short_blade_rotor_2',
    'Bird',
    'Bird+mini-helicopter_1',
    'Bird+mini-helicopter_2',
    'drone_1',
    'drone_2',
    'RC plane_1',
    'RC plane_2'
]

class MicroDopplerClassifier:
    def __init__(self, model_path="model_outputs/final_classification_model.keras"):
        """Initialize the classifier with model path"""
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            self.model = load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def is_micro_doppler_signature(self, image):
        """
        Enhanced validation to verify if the image is a micro-Doppler spectrogram
        Returns: (bool, str) - (is_valid, error_message)
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Check basic image properties
            if len(img_array.shape) not in [2, 3]:
                return False, "Invalid image format: Image must be 2D or 3D array"
            
            # Convert RGB to grayscale if needed
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 3:  # RGB
                    img_array = np.mean(img_array, axis=2)
                elif img_array.shape[2] == 4:  # RGBA
                    return False, "Invalid image format: RGBA images not supported"
            
            # Check image dimensions
            if img_array.shape[0] < 100 or img_array.shape[1] < 100:
                return False, "Image resolution too low for a spectrogram"
            
            # Spectrogram Characteristics Validation
            
            # 1. Check for expected intensity distribution
            hist, _ = np.histogram(img_array, bins=50)
            peaks = np.where(hist > np.mean(hist) + np.std(hist))[0]
            if len(peaks) < 2:  # Spectrograms typically have multiple intensity peaks
                return False, "Image lacks characteristic spectrogram intensity distribution"
            
            # 2. Check for temporal pattern
            time_variation = np.std(img_array, axis=1)
            if np.mean(time_variation) < 10:
                return False, "Image lacks temporal variation expected in spectrograms"
            
            # 3. Check for frequency content
            freq_variation = np.std(img_array, axis=0)
            if np.mean(freq_variation) < 10:
                return False, "Image lacks frequency variation expected in spectrograms"
            
            # 4. Check for typical spectrogram characteristics
            # - Spectrograms usually have structured patterns
            # - Should not be too uniform or too random
            block_size = 10
            blocks = img_array[:(img_array.shape[0]//block_size)*block_size, 
                              :(img_array.shape[1]//block_size)*block_size]
            blocks = blocks.reshape(blocks.shape[0]//block_size, block_size,
                                 blocks.shape[1]//block_size, block_size)
            block_std = np.std(blocks, axis=(1,3))
            
            if np.mean(block_std) < 5:  # Too uniform
                return False, "Image appears too uniform to be a spectrogram"
            
            if np.std(block_std) < 2:  # Too random
                return False, "Image lacks structured patterns expected in spectrograms"
            
            # 5. Check for common photo characteristics that shouldn't be in spectrograms
            edges_x = np.mean(np.abs(np.diff(img_array, axis=1)))
            edges_y = np.mean(np.abs(np.diff(img_array, axis=0)))
            if edges_x > 50 or edges_y > 50:  # Too many sharp edges
                return False, "Image contains too many sharp edges for a spectrogram"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error in signature validation: {str(e)}")
            return False, f"Error validating image: {str(e)}"

    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        try:
            # Resize image
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def classify(self, image, confidence_threshold=0.4):
        """
        Classify a micro-Doppler signature image
        Returns: dict with classification results and status
        """
        try:
            # First validate the image
            is_valid, error_message = self.is_micro_doppler_signature(image)
            
            if not is_valid:
                return {
                    'status': 'error',
                    'message': f"Invalid input: {error_message}",
                    'results': None
                }
            
            # Continue with classification only if validation passes
            img_array = self.preprocess_image(image)
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            results = []
            
            for idx in top_3_idx:
                confidence = float(predictions[0][idx])
                if confidence >= confidence_threshold:
                    results.append({
                        'category': CATEGORIES[idx],
                        'confidence': confidence * 100,
                        'is_threat': any(keyword in CATEGORIES[idx].lower() 
                                       for keyword in ['drone', 'rotor', 'rc plane'])
                    })
            
            if not results:
                return {
                    'status': 'uncertain',
                    'message': 'No classification met confidence threshold',
                    'results': None
                }
            
            return {
                'status': 'success',
                'message': 'Classification successful',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                'status': 'error',
                'message': f"Classification failed: {str(e)}",
                'results': None
            }

    def preprocess_image(self, image):
        """Preprocess the image for model input"""
        try:
            # Resize image
            image = image.resize((224, 224))
            
            # Convert to array and normalize
            img_array = img_to_array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def classify(self, image, confidence_threshold=0.4):
        """
        Classify a micro-Doppler signature image
        Returns: dict with classification results and status
        """
        try:
            # First validate the image
            is_valid, error_message = self.is_micro_doppler_signature(image)
            
            if not is_valid:
                return {
                    'status': 'error',
                    'message': error_message,
                    'results': None
                }
            
            # Preprocess the image
            img_array = self.preprocess_image(image)
            
            # Get predictions
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            results = []
            
            for idx in top_3_idx:
                confidence = float(predictions[0][idx])
                if confidence >= confidence_threshold:
                    results.append({
                        'category': CATEGORIES[idx],
                        'confidence': confidence * 100,
                        'is_threat': any(keyword in CATEGORIES[idx].lower() 
                                       for keyword in ['drone', 'rotor', 'rc plane'])
                    })
            
            if not results:
                return {
                    'status': 'uncertain',
                    'message': 'No classification met confidence threshold',
                    'results': None
                }
            
            return {
                'status': 'success',
                'message': 'Classification successful',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            return {
                'status': 'error',
                'message': f"Classification failed: {str(e)}",
                'results': None
            }

def main():
    """Test function for the classifier"""
    try:
        # Initialize classifier
        classifier = MicroDopplerClassifier()
        
        # Test with sample image
        test_image_path = input("Enter path to test image: ").strip()
        
        if not Path(test_image_path).exists():
            print("Error: Image file not found")
            return
        
        # Load and classify image
        image = Image.open(test_image_path)
        results = classifier.classify(image)
        
        # Print results
        print("\nClassification Results:")
        print(f"Status: {results['status']}")
        print(f"Message: {results['message']}")
        
        if results['results']:
            print("\nDetected Objects:")
            for idx, result in enumerate(results['results'], 1):
                print(f"\n{idx}. Category: {result['category']}")
                print(f"   Confidence: {result['confidence']:.2f}%")
                print(f"   Threat Status: {'⚠️ Potential Threat' if result['is_threat'] else '✅ Safe'}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()