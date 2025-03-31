import streamlit as st
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
AZURE_ENDPOINT = "http://e9cddab6-15c5-4a4d-bdee-6101f526ee75.eastus.azurecontainer.io/score"
AZURE_API_KEY = "<your-api-key>"

def send_image_to_azure(image_path):
    """Send image data to Azure endpoint."""
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
        "Content-Type": "application/octet-stream",  # Ensure binary data is sent
    }

    try:
        with open(image_path, "rb") as image_file:
            # Log the headers and file size for debugging
            logging.info(f"Sending image to Azure endpoint with headers: {headers}")
            logging.info(f"Image size: {os.path.getsize(image_path)} bytes")

            # Send the binary image data to the Azure endpoint
            response = requests.post(AZURE_ENDPOINT, headers=headers, data=image_file)

        if response.status_code == 200:
            logging.info("Image successfully sent to Azure endpoint.")
            return response.json()
        else:
            logging.error(f"Failed to send image. Status code: {response.status_code}, Response: {response.text}")
            st.error(f"Azure Error: {response.text}")  # Display Azure error in Streamlit
            return None
    except Exception as e:
        logging.error(f"An error occurred while sending the image: {str(e)}")
        st.error(f"An error occurred while sending the image: {str(e)}")
        return None

def main():
    st.title("Image Upload and Azure Processing")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image for processing", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Save the uploaded image temporarily
            temp_image_path = os.path.join("temp_image", uploaded_file.name)
            os.makedirs("temp_image", exist_ok=True)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"Image saved to {temp_image_path}.")

            with st.spinner("Sending image to Azure..."):
                # Send image to Azure
                response = send_image_to_azure(temp_image_path)

                # Display results
                if response:
                    st.image(temp_image_path, caption="Uploaded Image", use_column_width=True)
                    st.markdown("**Azure Response:**")
                    st.json(response)
                else:
                    st.error("Failed to process the image.")

            # Clean up temporary image
            os.remove(temp_image_path)
            logging.info(f"Temporary image {temp_image_path} deleted.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
