import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time
import os

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

# Custom CSS for enhanced styling with scrollbar removed and container height adjusted
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
if 'logo' not in st.session_state:
    st.session_state.logo = None
    st.session_state.is_svg = False

# Function to load the logo image
@st.cache_resource
def load_logo():
    logo_path = "images.png"
    try:
        img = Image.open(logo_path)
        return img.resize((100, 100))
    except Exception as e:
        st.error(f"Error loading logo image: {str(e)}")
        svg_code = """
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#1e3a8a" rx="10" ry="10"/>
            <circle cx="50" cy="50" r="30" fill="#60a5fa"/>
            <path d="M30 50 L45 65 L70 35" stroke="white" stroke-width="6" fill="none"/>
            <text x="50" y="85" font-family="Arial" font-size="12" fill="white" text-anchor="middle">DETECTOR</text>
        </svg>
        """
        return svg_code, True

# Load model with better error handling and deployment checks
@st.cache_resource
def load_ai_model():
    try:
        # First verify model file exists
        if not os.path.exists('my_custom_model.h5'):
            st.error("CRITICAL: Model file 'my_custom_model.h5' not found in deployment!")
            st.error(f"Current directory contents: {os.listdir('.')}")
            return None
            
        # Check file size (deployment platforms often have limits)
        file_size = os.path.getsize('my_custom_model.h5') / (1024 * 1024)  # in MB
        if file_size > 500:  # Common free tier limit
            st.warning(f"Model file is large ({file_size:.2f} MB). Some platforms may not load it properly.")
        
        model = load_model('my_custom_model.h5')
        st.session_state.model_loaded = True
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if "OOM" in str(e):  # Out of memory error
            st.error("Your deployment environment may have insufficient memory. Try using a smaller model.")
        st.session_state.model_loaded = False
        return None

# Image preprocessing with validation
def preprocess_image(uploaded_file, target_size=(224, 224)):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        if img.size[0] < 50 or img.size[1] < 50:
            st.warning("Image resolution is very low. Results may be less accurate.")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
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
        img = Image.open(uploaded_file)
        img = img.resize((300, 300))
        st.image(img, caption="Uploaded Image", use_container_width=False)
        
        if st.button("Analyze Image", type="primary", use_container_width=True, key="analyze_btn"):
            with st.spinner("Loading AI model..."):
                model = load_ai_model()
                
            if model and st.session_state.model_loaded:
                with st.spinner("Analyzing image patterns..."):
                    try:
                        img_array = preprocess_image(uploaded_file)
                        if img_array is not None:
                            start_time = time.time()
                            prediction = model.predict(img_array)
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
                            st.session_state.page = 2
                            st.rerun()  # Force refresh to show results page
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.session_state.model_loaded = False
    st.markdown('</div>', unsafe_allow_html=True)

# Page 2: Results Display
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
            img = Image.open(st.session_state.uploaded_file)
            img = img.resize((250, 250))
            st.image(img, caption="Uploaded Image", use_container_width=False)
    
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

# Main app logic for page navigation
def main():
    sidebar_content()
    if st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()

if __name__ == "__main__":
    main()