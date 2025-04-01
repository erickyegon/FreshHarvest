"""
FreshHarvest Streamlit App
==========================
A modern, interactive web application for fruit freshness detection.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
from PIL import Image
import io
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Try to import the FreshHarvest modules with error handling
try:
    from config import DEVICE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
    from inference import FruitFreshnessInference
    from utils import get_timestamp_str

    modules_imported = True
except ImportError as e:
    modules_imported = False
    import_error = str(e)

# Set page configuration
st.set_page_config(
    page_title="FreshHarvest - Fruit Freshness Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C8C3C;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #333;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fresh-box {
        background-color: rgba(76, 175, 80, 0.1);
        border: 2px solid #4CAF50;
    }
    .spoiled-box {
        background-color: rgba(244, 67, 54, 0.1);
        border: 2px solid #F44336;
    }
    .prediction-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .fresh-header {
        color: #2E7D32;
    }
    .spoiled-header {
        color: #C62828;
    }
    .confidence-meter {
        height: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .confidence-value {
        font-size: 1.1rem;
        font-weight: 500;
        text-align: center;
    }
    .metrics-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f5f5f5;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-container:hover {
        border-color: #2C8C3C;
    }
    .explanation {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .fruit-type {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 0.5rem;
        color: #333;
    }
    .stProgress > div > div > div > div {
        background-color: #2C8C3C;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFEBEE;
        border: 1px solid #FFCDD2;
        color: #B71C1C;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border: 1px solid #BBDEFB;
        color: #0D47A1;
        margin: 1rem 0;
    }
    .sample-image-container {
        border: 1px solid #ddd;
        border-radius: 0.25rem;
        padding: 0.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .sample-image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sample-image-title {
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    /* Style for file uploader */
    .stFileUploader>div>button {
        background-color: #2C8C3C;
        color: white;
    }
    .stFileUploader>div>button:hover {
        background-color: #1e7e34;
    }
    .drag-drop-text {
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .drag-drop-subtext {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """
    Load the model from path with caching for efficiency

    Args:
        model_path (str): Path to the model

    Returns:
        FruitFreshnessInference: Loaded inference model or None if error
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None

        inference_model = FruitFreshnessInference(model_path)
        return inference_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(uploaded_file):
    """
    Preprocess the uploaded image

    Args:
        uploaded_file: Uploaded file from Streamlit

    Returns:
        PIL.Image: Processed image or None if error
    """
    try:
        # Read image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def detect_fruit_type(image):
    """
    Detect the fruit type from the image using image analysis
    This is a simple implementation - in a production system,
    this would be handled by a dedicated fruit type classifier

    Args:
        image (PIL.Image): Image to detect fruit type from

    Returns:
        str: Detected fruit type
    """
    try:
        # Convert to numpy array for OpenCV
        img_array = np.array(image)

        # Convert to HSV color space
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

        # Calculate color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)

        # Get dominant hue and saturation
        dominant_h = np.argmax(h_hist)
        dominant_s = np.argmax(s_hist)

        # Simple color-based fruit type detection
        if 0 <= dominant_h <= 30 or 150 <= dominant_h <= 180:  # Red hues
            if dominant_s > 150:
                return "Strawberry"
            else:
                return "Tomato"
        elif 30 <= dominant_h <= 60:  # Yellow/Orange hues
            if dominant_s > 150:
                return "Mango"
            else:
                return "Banana"
        elif 60 <= dominant_h <= 90:  # Green-Yellow hues
            return "Lemon"
        elif 90 <= dominant_h <= 120:  # Green hues
            return "Lulo"
        elif 0 <= dominant_h <= 20:  # Orange hues
            return "Orange"
        else:
            return "Unknown"
    except Exception as e:
        # Return unknown if any error occurs
        return "Unknown"


def show_prediction(image, prediction_result, fruit_type=None):
    """
    Display the prediction with visualization

    Args:
        image (PIL.Image): Original image
        prediction_result (dict): Prediction result from model
        fruit_type (str, optional): Detected fruit type
    """
    # Extract prediction details
    predicted_class = prediction_result['prediction']
    confidence = prediction_result['confidence']
    inference_time = prediction_result['inference_time']

    # Display image and prediction in columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Fixed: Replaced use_column_width with use_container_width
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if fruit_type and fruit_type != "Unknown":
            st.markdown(f"<div class='fruit-type'>Detected Fruit Type: {fruit_type}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='metrics-container'>Inference Time: {inference_time * 1000:.2f} ms</div>",
                    unsafe_allow_html=True)

    with col2:
        # Show prediction box based on class
        if predicted_class == "fresh":
            st.markdown(f"""
            <div class="prediction-box fresh-box">
                <div class="prediction-header fresh-header">FRESH</div>
                <div class="confidence-value">Confidence: {confidence * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box spoiled-box">
                <div class="prediction-header spoiled-header">SPOILED</div>
                <div class="confidence-value">Confidence: {confidence * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Display confidence meter
        st.markdown("<div class='confidence-meter'>", unsafe_allow_html=True)
        st.progress(confidence)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show explanation
        explanation = get_prediction_explanation(predicted_class, confidence, fruit_type)
        st.markdown(f"""
        <div class="explanation">
            <h3>Explanation</h3>
            <p>{explanation}</p>
        </div>
        """, unsafe_allow_html=True)

    # Show detailed probabilities if available
    if 'probabilities' in prediction_result:
        st.subheader("Classification Probabilities")

        # Create DataFrame for display
        probs_df = pd.DataFrame({
            'Class': list(prediction_result['probabilities'].keys()),
            'Probability': list(prediction_result['probabilities'].values())
        })

        try:
            # Plot probabilities
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.barplot(x='Probability', y='Class', data=probs_df,
                        palette=['#4CAF50' if c == 'fresh' else '#F44336' for c in probs_df['Class']])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_ylabel('Class')
            ax.set_title('Classification Probabilities')

            # Fixed: Replaced with a direct function call without parameters
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating probability chart: {e}")
            # Display as text instead
            st.dataframe(probs_df)


def get_prediction_explanation(predicted_class, confidence, fruit_type=None):
    """
    Generate an explanation for the prediction

    Args:
        predicted_class (str): Predicted class (fresh or spoiled)
        confidence (float): Prediction confidence
        fruit_type (str, optional): Detected fruit type

    Returns:
        str: Explanation text
    """
    if predicted_class == "fresh":
        if confidence > 0.9:
            base_text = "This fruit appears to be very fresh."
        elif confidence > 0.7:
            base_text = "This fruit is likely fresh, but may be getting close to its peak ripeness."
        else:
            base_text = "This fruit appears to be fresh, but the model is less certain. It might be approaching the end of its freshness period."
    else:
        if confidence > 0.9:
            base_text = "This fruit shows clear signs of spoilage."
        elif confidence > 0.7:
            base_text = "This fruit is likely spoiled, showing some indicators of decay."
        else:
            base_text = "This fruit may be starting to spoil, but the model is less certain. It might be at the early stages of spoilage."

    # Add fruit-specific guidance if available
    if fruit_type and fruit_type != "Unknown":
        if fruit_type == "Banana":
            if predicted_class == "fresh":
                return base_text + " Fresh bananas typically have bright yellow skin with minimal brown spots."
            else:
                return base_text + " Spoiled bananas often have extensive brown or black patches and may feel mushy."

        elif fruit_type == "Strawberry":
            if predicted_class == "fresh":
                return base_text + " Fresh strawberries have vibrant red color without soft spots or mold."
            else:
                return base_text + " Spoiled strawberries often have dark spots, mold, or a mushy texture."

        elif fruit_type == "Tomato":
            if predicted_class == "fresh":
                return base_text + " Fresh tomatoes have firm skin with consistent color and no wrinkles or soft spots."
            else:
                return base_text + " Spoiled tomatoes may have wrinkled skin, leaking juice, or visible mold."

        elif fruit_type == "Lemon":
            if predicted_class == "fresh":
                return base_text + " Fresh lemons have bright yellow skin without dark spots or mold."
            else:
                return base_text + " Spoiled lemons often have soft spots, discoloration, or visible mold growth."

        elif fruit_type == "Orange":
            if predicted_class == "fresh":
                return base_text + " Fresh oranges have firm, vibrant skin without soft spots or wrinkles."
            else:
                return base_text + " Spoiled oranges may have soft areas, mold, or discoloration."

        elif fruit_type == "Mango":
            if predicted_class == "fresh":
                return base_text + " Fresh mangoes have firm flesh with smooth, unblemished skin."
            else:
                return base_text + " Spoiled mangoes often have dark spots, overly soft texture, or fermented smell."

    return base_text


def create_sample_images_section():
    """Create a section showing sample fruit images"""
    st.subheader("Sample Fruits")

    # Create two rows of fruits
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

    # Define placeholder texts
    fruits = [
        {"name": "Banana", "description": "Yellow, curved fruit with a soft interior"},
        {"name": "Strawberry", "description": "Red berries with seeds on the exterior"},
        {"name": "Orange", "description": "Round citrus fruit with orange peel"},
        {"name": "Lemon", "description": "Yellow citrus fruit with sour taste"},
        {"name": "Mango", "description": "Sweet tropical fruit with orange flesh"},
        {"name": "Tomato", "description": "Red fruit often used as a vegetable"},
        {"name": "Lulo", "description": "Orange tropical fruit related to tomato"},
        {"name": "Tamarillo", "description": "Egg-shaped fruit with red skin"}
    ]

    # First row
    with row1_col1:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[0]['name']}</div>
            <div style="width:100%;height:120px;background:#FFD54F;display:flex;justify-content:center;align-items:center;color:#5D4037;">
                {fruits[0]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[0]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col2:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[1]['name']}</div>
            <div style="width:100%;height:120px;background:#E57373;display:flex;justify-content:center;align-items:center;color:#FFFFFF;">
                {fruits[1]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[1]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col3:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[2]['name']}</div>
            <div style="width:100%;height:120px;background:#FF9800;display:flex;justify-content:center;align-items:center;color:#5D4037;">
                {fruits[2]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[2]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row1_col4:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[3]['name']}</div>
            <div style="width:100%;height:120px;background:#FFEB3B;display:flex;justify-content:center;align-items:center;color:#5D4037;">
                {fruits[3]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[3]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Second row
    with row2_col1:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[4]['name']}</div>
            <div style="width:100%;height:120px;background:#FFA726;display:flex;justify-content:center;align-items:center;color:#5D4037;">
                {fruits[4]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[4]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col2:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[5]['name']}</div>
            <div style="width:100%;height:120px;background:#F44336;display:flex;justify-content:center;align-items:center;color:#FFFFFF;">
                {fruits[5]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[5]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col3:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[6]['name']}</div>
            <div style="width:100%;height:120px;background:#FF7043;display:flex;justify-content:center;align-items:center;color:#5D4037;">
                {fruits[6]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[6]['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with row2_col4:
        st.markdown(f"""
        <div class="sample-image-container">
            <div class="sample-image-title">{fruits[7]['name']}</div>
            <div style="width:100%;height:120px;background:#E53935;display:flex;justify-content:center;align-items:center;color:#FFFFFF;">
                {fruits[7]['name']}
            </div>
            <p style="font-size:0.8rem;">{fruits[7]['description']}</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function for the Streamlit app"""
    # Display header
    st.markdown("<div class='main-header'>FreshHarvest</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Fruit Freshness Detection</div>", unsafe_allow_html=True)

    # Check if the required modules were imported successfully
    if not modules_imported:
        st.markdown(f"""
        <div class="error-box">
            <h3>Error Loading Modules</h3>
            <p>There was an error loading the required modules: {import_error}</p>
            <p>Please ensure you're running the app from the correct directory with all dependencies installed.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show simplified interface in case of module error
        st.markdown("### Upload a Fruit Image")
        st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],
                         disabled=True, help="Module loading error - functionality disabled")

        # Add some helpful information
        st.markdown("""
        #### How to fix this issue:
        1. Make sure you've installed all dependencies: `pip install -r requirements.txt`
        2. Run the app from the project root directory: `streamlit run app_streamlit.py`
        3. Check that all required Python modules are installed and accessible
        """)

        # Display footer and exit
        st.markdown("""
        <div class="footer">
            <p>© 2025 FreshHarvest - Fruit Freshness Detection | Built with Streamlit and PyTorch</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        **FreshHarvest** is an AI-powered system that detects whether fruits are fresh or spoiled using computer vision and deep learning.

        ### How it works
        1. Upload an image of a fruit
        2. The AI model analyzes the image
        3. The app displays whether the fruit is fresh or spoiled

        ### Supported Fruits
        - Banana
        - Lemon
        - Lulo
        - Mango
        - Orange
        - Strawberry
        - Tamarillo
        - Tomato
        """)

        st.divider()

        # Model selection
        st.subheader("Model Selection")
        model_path = st.text_input(
            "Model Path",
            value="model_outputs/final_model.pt",
            help="Path to the trained model file (.pt)"
        )

        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                inference_model = load_model(model_path)
                if inference_model:
                    st.session_state['model'] = inference_model
                    st.success("Model loaded successfully!")

    # Main content
    # Check if model is loaded
    if 'model' not in st.session_state:
        # Auto-load the model on first run
        with st.spinner("Loading default model..."):
            default_model_path = "model_outputs/final_model.pt"
            inference_model = load_model(default_model_path)
            if inference_model:
                st.session_state['model'] = inference_model
                st.success("Default model loaded successfully!")
            else:
                st.warning(f"Default model not found at {default_model_path}.")
                # Create a friendly error message with instructions
                st.markdown(f"""
                <div class="info-box">
                    <h3>Model Not Found</h3>
                    <p>The default model could not be found at <code>{default_model_path}</code>.</p>
                    <p>Please ensure you've:</p>
                    <ol>
                        <li>Trained a model using <code>python main.py --mode train</code></li>
                        <li>Confirmed the model exists at the expected location</li>
                        <li>Or specify a different model path in the sidebar</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)

                # Continue with limited functionality
                st.session_state['model'] = None

    # Display upload options
    st.subheader("Upload a Fruit Image")

    # Add text above the uploader to encourage drag-and-drop
    st.markdown("""
    <p class="drag-drop-text">Drag and drop a fruit image here</p>
    <p class="drag-drop-subtext">Or click below to browse files</p>
    """, unsafe_allow_html=True)

    # File uploader - this automatically supports drag-and-drop in Streamlit
    uploaded_file = st.file_uploader(
        "Upload Fruit Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="fruit_image",
        accept_multiple_files=False,
        help="Drag and drop a fruit image here or click to browse"
    )

    # Process the uploaded image
    if uploaded_file is not None:
        if 'model' in st.session_state and st.session_state['model'] is not None:
            # Display a spinner while processing
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                image = preprocess_image(uploaded_file)

                if image:
                    try:
                        # Detect fruit type (this is a simple implementation)
                        fruit_type = detect_fruit_type(image)

                        # Make prediction
                        prediction_result = st.session_state['model'].predict(image, return_probs=True)

                        # Show prediction
                        show_prediction(image, prediction_result, fruit_type)
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        # Fixed: Replaced use_column_width with use_container_width
                        st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.warning("Model not loaded. Please load a model from the sidebar first.")
            # Fixed: Replaced use_column_width with use_container_width
            st.image(Image.open(uploaded_file), caption="Uploaded Image (Not Analyzed)", use_container_width=True)

    # Instructions when no image is uploaded
    elif 'model' in st.session_state and st.session_state['model'] is not None:
        st.info("Upload an image of a fruit to detect its freshness")
        # Show sample images with colored placeholders instead of missing images
        create_sample_images_section()

    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 FreshHarvest - Fruit Freshness Detection | Built with Streamlit and PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()