import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time
import os
import logging

# Set up logging to debug deployment issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config with wider layout and better theme
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Deepfake Detection Tool\nDetect AI-generated images with deep learning"
    }
)

# Keep your existing CSS (unchanged)
st.markdown("""
<style>
    body, .main {
        background-color: #111827;
        color: #f9fafb;
    }
    
    /* Hide main scrollbar */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        overflow: visible !important;
        max-height: none !important;
    }
    
    /* Hide scrollbar for Chrome, Safari and Opera */
    .main::-webkit-scrollbar {
        display: none;
    }
    
    /* Hide scrollbar for IE, Edge and Firefox */
    .main {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }

    .header {
        padding: 1rem 0;
        border-bottom: 1px solid #374151;
        margin-bottom: 1.5rem;
    }

    .upload-card, .results-card, .stat-card, .sidebar-card {
        background: #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        color: #f9fafb;
        margin-bottom: 1.5rem;
    }

    .confidence-meter {
        height: 15px;
        border-radius: 10px;
        background: #374151;
        margin: 1rem 0;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    .stButton>button {
        background-color: #2563eb;
        color: #f9fafb;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
        margin: 0.5rem;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
    }

    .stFileUploader>div>div>div>div {
        border: 2px dashed #6b7280;
        border-radius: 8px;
        padding: 1rem;
        color: #f9fafb;
    }

    [data-testid="stSidebar"] {
        background-color: #1f2937;
        padding: 1rem;
    }

    .sidebar-card h3, .sidebar-card p, .sidebar-card li, .step-circle span {
        color: #f9fafb !important;
    }

    .step-circle {
        background: #2563eb;
        color: white;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.5rem;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, ul, li {
        color: #f9fafb !important;
    }

    .css-1d391kg, .css-1cpxqw2, .css-ffhzg2, .css-12oz5g7 {
        color: #f9fafb !important;
    }
    
    .logo-img {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border: 2px solid #2563eb;
        margin-top: 15px;
    }
    
    /* Container for logo to properly center and add padding */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 20px;
        height: 100%;
    }
    
    /* Adjust content spacing for better fit */
    .stImage {
        margin-bottom: 1rem;
    }
    
    /* Adjust expander styling */
    .streamlit-expanderHeader {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border-radius: 8px !important;
    }
    
    /* Make sure content is properly spaced */
    .block-container > div > div {
        gap: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize all session state variables at the start
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'error_msg' not in st.session_state:
    st.session_state.error_msg = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'logo' not in st.session_state:
    st.session_state.logo = None
    st.session_state.is_svg = False

# Function to load the logo image
@st.cache_resource
def load_logo():
    # Try multiple possible paths for logo
    possible_paths = ["images.png", "./images.png", "../images.png", "/app/images.png"]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                logger.info(f"Found logo at {path}")
                img = Image.open(path)
                return img.resize((100, 100))
        except Exception as e:
            logger.warning(f"Failed to load logo from {path}: {str(e)}")
            continue
    
    # Fallback to SVG
    logger.info("Using SVG fallback for logo")
    svg_code = """
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#1e3a8a" rx="10" ry="10"/>
        <circle cx="50" cy="50" r="30" fill="#60a5fa"/>
        <path d="M30 50 L45 65 L70 35" stroke="white" stroke-width="6" fill="none"/>
        <text x="50" y="85" font-family="Arial" font-size="12" fill="white" text-anchor="middle">DETECTOR</text>
    </svg>
    """
    return svg_code, True

# Load model with improved error handling for deployment environments
@st.cache_resource
def load_ai_model():
    # Check if model is already loaded in session state
    if st.session_state.model is not None and st.session_state.model_loaded:
        logger.info("Using already loaded model from session state")
        return st.session_state.model
    
    # Try multiple possible paths for model
    possible_paths = ["my_custom_model.h5", "./my_custom_model.h5", "../my_custom_model.h5", "/app/my_custom_model.h5"]
    
    for model_path in possible_paths:
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
                continue
                
            # Check file size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB
            logger.info(f"Model file size: {file_size:.2f} MB")
            
            # Load with reduced memory usage (may help with deployment constraints)
            import tensorflow as tf
            tf_version = tf.__version__
            logger.info(f"TensorFlow version: {tf_version}")
            
            # Try to limit memory growth to avoid OOM errors
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                if len(physical_devices) > 0:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    logger.info("Set GPU memory growth")
            except:
                logger.warning("Failed to set GPU memory growth")
            
            # Load the model
            model = load_model(model_path)
            
            # Test with a dummy tensor to ensure model works
            try:
                dummy_data = np.zeros((1, 224, 224, 3))
                _ = model.predict(dummy_data, verbose=0)
                logger.info("Model successfully tested with dummy data")
            except Exception as e:
                logger.error(f"Model loaded but failed with dummy input: {str(e)}")
                continue
            
            # Success - save model to session state
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.error_msg = None
            logger.info(f"Model successfully loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            st.session_state.error_msg = str(e)
            continue
    
    # If we get here, we couldn't load the model from any path
    st.session_state.model_loaded = False
    st.error("Failed to load model from any location. Please check deployment configuration.")
    logger.error("Failed to load model from any location")
    return None

# Image preprocessing with validation and better error handling
def preprocess_image(uploaded_file, target_size=(224, 224)):
    try:
        logger.info(f"Processing uploaded file: {uploaded_file.name}, size: {uploaded_file.size} bytes")
        
        # Read image into memory
        image_bytes = uploaded_file.getvalue()
        logger.info(f"Successfully read {len(image_bytes)} bytes from uploaded file")
        
        # Open image with PIL
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Original image size: {img.size}")
        
        # Validate image
        if img.size[0] < 50 or img.size[1] < 50:
            st.warning("Image resolution is very low. Results may be less accurate.")
        
        # Resize and normalize
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        
        # Verify array shape
        logger.info(f"Preprocessed image array shape: {img_array.shape}")
        
        # Expand dimensions for batch
        expanded_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Final input shape: {expanded_array.shape}")
        
        return expanded_array
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"Error processing image: {str(e)}")
        return None

# Load logo if not already loaded
if st.session_state.logo is None:
    logo_result = load_logo()
    if isinstance(logo_result, tuple):
        st.session_state.logo = logo_result[0]
        st.session_state.is_svg = True
    else:
        st.session_state.logo = logo_result
        st.session_state.is_svg = False

# Sidebar content
def sidebar_content():
    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### How It Works")
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div class="step-circle">1</div>
            <span>Upload any image file</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div class="step-circle">2</div>
            <span>Analyze with our AI model</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div class="step-circle">3</div>
            <span>Get detailed authenticity report</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### About the Technology")
        st.markdown("""
        <p>Our detection model uses deep convolutional neural networks trained on thousands of AI-generated and real images to identify subtle patterns that distinguish synthetic from authentic images.</p>
        <p>The model analyzes:</p>
        <ul>
            <li>Texture patterns</li>
            <li>Color distribution</li>
            <li>Edge consistency</li>
            <li>Noise characteristics</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add deployment status indicators
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### System Status")
        
        if st.session_state.model_loaded:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not loaded")
            if st.session_state.error_msg:
                with st.expander("Error details"):
                    st.code(st.session_state.error_msg)
        
        # Add system info
        import platform
        st.markdown(f"**Platform:** {platform.system()}")
        st.markdown(f"**Python:** {platform.python_version()}")
        try:
            import tensorflow as tf
            st.markdown(f"**TensorFlow:** {tf.__version__}")
        except:
            st.markdown("**TensorFlow:** Not found")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Page 1: Image Upload
def page_1():
    # Header section with improved spacing
    st.markdown('<div class="header" style="display: flex; align-items: center; gap: 30px; padding: 1.5rem 0;">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 3])
    with col1:
        if st.session_state.is_svg:
            st.markdown(f'<div class="logo-container"><div class="logo-img">{st.session_state.logo}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image(st.session_state.logo, width=100, output_format="AUTO", clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="padding-top: 0.8rem;">', unsafe_allow_html=True)
        st.title("Deepfake Detection")
        st.markdown("Detect whether an image was generated by AI or captured by a camera.")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, JPEG)", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        try:
            img = Image.open(uploaded_file)
            img = img.resize((300, 300))
            st.image(img, caption="Uploaded Image", use_container_width=False)
            
            if st.button("Analyze Image", type="primary", use_container_width=True, key="analyze_btn"):
                logger.info("Analyze button clicked")
                
                # Pre-load model to avoid waiting later
                with st.spinner("Loading AI model..."):
                    model = load_ai_model()
                    
                if model and st.session_state.model_loaded:
                    with st.spinner("Analyzing image patterns..."):
                        try:
                            # Reset file position to beginning before reading
                            uploaded_file.seek(0)
                            
                            # Process image
                            img_array = preprocess_image(uploaded_file)
                            logger.info(f"Image preprocessing successful: {img_array is not None}")
                            
                            if img_array is not None:
                                # Add some debug info
                                st.info(f"Image shape: {img_array.shape}, Min: {img_array.min()}, Max: {img_array.max()}")
                                
                                # Make prediction with proper error handling
                                try:
                                    start_time = time.time()
                                    logger.info("Starting model prediction")
                                    
                                    # Use a timeout to prevent hanging
                                    import threading
                                    import queue
                                    
                                    def predict_with_timeout(model, img_array, result_queue):
                                        try:
                                            pred = model.predict(img_array, verbose=1)
                                            result_queue.put(pred)
                                        except Exception as e:
                                            result_queue.put(e)
                                    
                                    # Create a queue for the result
                                    q = queue.Queue()
                                    
                                    # Create and start the prediction thread
                                    thread = threading.Thread(target=predict_with_timeout, args=(model, img_array, q))
                                    thread.start()
                                    
                                    # Wait for the thread to finish with timeout
                                    thread.join(timeout=30)  # 30 seconds timeout
                                    
                                    if thread.is_alive():
                                        # If thread is still running after timeout
                                        logger.error("Prediction timed out after 30 seconds")
                                        st.error("Analysis timed out. The model may be overloaded.")
                                        # Don't proceed
                                        return
                                    
                                    # Get the result from the queue
                                    result = q.get()
                                    if isinstance(result, Exception):
                                        # If an exception was put in the queue
                                        raise result
                                    
                                    prediction = result
                                    logger.info(f"Prediction completed: {prediction}")
                                    
                                    inference_time = time.time() - start_time
                                    
                                    is_real = prediction[0][0] > 0.5
                                    label = "Real Photo" if is_real else "AI-Generated"
                                    confidence = min(100, (prediction[0][0] if is_real else 1 - prediction[0][0]) * 100)
                                    confidence_color = "#10b981" if is_real else "#ef4444"
                                    
                                    st.session_state.prediction_result = {
                                        'label': label,
                                        'confidence': confidence,
                                        'confidence_color': confidence_color,
                                        'inference_time': inference_time,
                                        'is_real': is_real
                                    }
                                    
                                    logger.info(f"Analysis complete: {label} with {confidence:.1f}% confidence")
                                    st.session_state.page = 2
                                    st.rerun()  # Force refresh to show results page
                                    
                                except Exception as e:
                                    logger.error(f"Model prediction failed: {str(e)}")
                                    st.error(f"Model prediction failed: {str(e)}")
                                    # Add more specific error handling
                                    if "shape" in str(e).lower():
                                        st.error("Input shape mismatch. Please try another image.")
                                    elif "memory" in str(e).lower():
                                        st.error("Out of memory error. The deployment environment may have insufficient resources.")
                                    elif "timeout" in str(e).lower():
                                        st.error("Operation timed out. The server may be overloaded.")
                        except Exception as e:
                            logger.error(f"Analysis failed: {str(e)}")
                            st.error(f"Analysis failed: {str(e)}")
                else:
                    st.error("Model could not be loaded. Please check system status in the sidebar.")
                    # Show a more helpful diagnostic message
                    st.error("Deployment environment may not have the required resources or permissions to load the model.")
        except Exception as e:
            logger.error(f"Error displaying uploaded image: {str(e)}")
            st.error(f"Error displaying uploaded image: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Results Display (same as original but with better error handling)
def page_2():
    st.markdown('<div class="header" style="display: flex; align-items: center; gap: 30px; padding: 1.5rem 0;">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 3])
    with col1:
        if st.session_state.is_svg:
            st.markdown(f'<div class="logo-container"><div class="logo-img">{st.session_state.logo}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image(st.session_state.logo, width=100, output_format="AUTO", clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="padding-top: 0.8rem;">', unsafe_allow_html=True)
        st.title("Deepfake Detection")
        st.markdown("Analysis Results")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_img, col_space = st.columns([1, 2])
    with col_img:
        if st.session_state.uploaded_file is not None:
            try:
                # Reset file position to beginning before reading
                st.session_state.uploaded_file.seek(0)
                img = Image.open(st.session_state.uploaded_file)
                img = img.resize((250, 250))
                st.image(img, caption="Uploaded Image", use_container_width=False)
            except Exception as e:
                logger.error(f"Error displaying result image: {str(e)}")
                st.error("Unable to display image. The file may be corrupted or no longer available.")
    
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.markdown('<div class="results-card">', unsafe_allow_html=True)
        st.subheader("Analysis Results")
        
        st.markdown(f"""
        <div style="display: inline-block; padding: 0.5rem 1rem; 
            background-color: {result['confidence_color']}20; 
            border-radius: 8px; 
            border-left: 4px solid {result['confidence_color']};
            margin-bottom: 1rem;">
            <h3 style="color: {result['confidence_color']}; margin: 0;">{result['label']}</h3>
            <p style="margin: 0; color: #6b7280;">Confidence: {result['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {result['confidence']}%; background: {result['confidence_color']};"></div>
        </div>
        """, unsafe_allow_html=True)
        
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Prediction", result['label'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with stats_cols[1]:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{result['confidence']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with stats_cols[2]:
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            st.metric("Processing Time", f"{result['inference_time']:.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Technical Analysis Details", expanded=False):
            if result['is_real']:
                st.markdown("""
                **Natural Image Characteristics:**
                - Organic noise distribution
                - Consistent lighting
                - Natural depth of field
                - Realistic texture variations
                - Authentic imperfections
                """)
            else:
                st.markdown("""
                **AI Generation Indicators Detected:**
                - Unnatural texture patterns
                - Overly smooth surfaces
                - Inconsistent lighting/shadow
                - Repetitive elements
                - Artifacts in fine details
                """)
            
            st.caption("Note: No detection method is 100% accurate. Results should be considered as probabilistic estimates.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        if st.button("Back to Upload", use_container_width=True):
            st.session_state.page = 1
            st.rerun()
    with btn_cols[1]:
        if st.button("Start Over", use_container_width=True):
            st.session_state.page = 1
            st.session_state.uploaded_file = None
            st.session_state.prediction_result = None
            st.rerun()

# Main app logic with better error handling
def main():
    try:
        sidebar_content()
        if st.session_state.page == 1:
            page_1()
        elif st.session_state.page == 2:
            page_2()
    except Exception as e:
        logger.error(f"Unexpected error in main app: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        # Add a way to recover
        st.button("Reset Application", on_click=lambda: st.session_state.clear())

if __name__ == "__main__":
    main()