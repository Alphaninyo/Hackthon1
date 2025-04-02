import streamlit as st
import os
import requests
import logging
import numpy as np
from PIL import Image
import json
from skimage.feature import hog
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
AZURE_ENDPOINT = "http://6e0448db-282e-4f7c-b8b6-2b4f8e147e6b.eastus.azurecontainer.io/score"

def extract_features(image_path):
    """Extract HOG features to match training preprocessing"""
    try:
        # Load and standardize image size (matching training)
        img = load_img(image_path, target_size=(256, 256))
        
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        
        # Convert to grayscale for HOG
        img_gray = rgb2gray(img_array)
        
        # Extract HOG features (matching training parameters)
        features = hog(img_gray,
                      orientations=8,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      transform_sqrt=True)
        
        return features
        
    except Exception as e:
        logging.error(f"Feature extraction error: {str(e)}")
        raise

def validate_endpoint():
    """Validate if the Azure endpoint is accessible."""
    try:
        headers = {"Content-Type": "application/json"}
        # Create sample data matching the expected feature size
        sample_features = np.zeros(30752)  # HOG feature size from training
        test_data = json.dumps({"data": [sample_features.tolist()]})
        response = requests.post(AZURE_ENDPOINT, headers=headers, data=test_data)
        
        logging.info(f"Endpoint validation status code: {response.status_code}")
        if response.status_code == 200:
            return True
        else:
            st.error(f"Endpoint validation failed with status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Azure endpoint is not accessible: {str(e)}")
        logging.error(f"Azure endpoint connection error: {str(e)}")
        return False

def send_image_to_azure(image_path):
    """Send processed image data to Azure endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        # Extract HOG features
        processed_data = extract_features(image_path)
        
        # Create payload with processed features
        payload = {
            "data": [processed_data.tolist()]
        }

        logging.info(f"Sending request to Azure...")
        response = requests.post(
            AZURE_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )

        logging.info(f"Raw response text: {response.text}")

        if response.status_code == 200:
            try:
                # Parse the response
                result = json.loads(response.text)  # Parse text directly
                logging.info(f"Parsed result type: {type(result)}")
                logging.info(f"Parsed result: {result}")
                
                # Map numerical predictions to labels
                label_map = {0: "Organic", 1: "Recyclable"}
                
                # Extract prediction directly from result structure
                if isinstance(result, dict) and 'result' in result and isinstance(result['result'], list):
                    prediction = int(float(result['result'][0]))
                    logging.info(f"Raw prediction value: {prediction}")
                    
                    predicted_label = label_map.get(prediction, "Unknown")
                    logging.info(f"Predicted label: {predicted_label}")
                    return {
                        "prediction": predicted_label,
                        "raw_value": prediction,
                        "confidence": "High"
                    }
                
                logging.error(f"Unexpected result structure: {result}")
                return {"prediction": "Error: Invalid result structure"}
                    
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                return {"prediction": "Error: Invalid JSON format"}
            except Exception as e:
                logging.error(f"Error processing response: {str(e)}")
                return {"prediction": f"Error: {str(e)}"}
        else:
            logging.error(f"Request failed with status {response.status_code}")
            return {"prediction": f"Error: Request failed ({response.status_code})"}
            
    except Exception as e:
        logging.error(f"Request failed: {str(e)}")
        return {"prediction": f"Error: {str(e)}"}

def main():
    st.title("Image Upload and Azure Processing")

    uploaded_file = st.file_uploader("Upload an image for processing", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            temp_image_path = os.path.join("temp_image", uploaded_file.name)
            os.makedirs("temp_image", exist_ok=True)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"Image saved to {temp_image_path}.")

            with st.spinner("Sending image to Azure..."):
                response = send_image_to_azure(temp_image_path)

                if response:
                    st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)
                    st.markdown("**Azure Response:**")
                    st.json(response)
                else:
                    st.error("Failed to process the image.")

            os.remove(temp_image_path)
            logging.info(f"Temporary image {temp_image_path} deleted.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
