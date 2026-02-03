"""
MNIST Digit Recognition - Streamlit App
Draw a digit and get real-time predictions!
"""

import streamlit as st
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(145deg, #1e2a4a, #243b55);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .big-prediction {
        font-size: 14rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin: 0;
    }
    
    .confidence-text {
        color: #10b981;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .label-text {
        color: #94a3b8;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    /* Instructions card */
    .instructions-card {
        background: rgba(30, 42, 74, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .instructions-card h3 {
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .instructions-card ul {
        color: #94a3b8;
    }
    
    /* Canvas container */
    .canvas-container {
        background: linear-gradient(145deg, #1e2a4a, #243b55);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    try:
        with open('mnist_logistic_regression_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'mnist_logistic_regression_model.pkl' is in the same directory.")
        return None

def preprocess_canvas_image(canvas_result):
    """Convert canvas drawing to MNIST format (28x28 grayscale) with proper centering"""
    if canvas_result.image_data is None:
        return None
    
    # Get the image data
    img_data = canvas_result.image_data
    
    # Convert to PIL Image
    img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img_gray)
    
    # Find bounding box of the digit (non-zero pixels)
    non_zero = np.where(img_array > 20)  # threshold to remove noise
    
    if len(non_zero[0]) == 0:
        return None
    
    # Get bounding box
    top, bottom = non_zero[0].min(), non_zero[0].max()
    left, right = non_zero[1].min(), non_zero[1].max()
    
    # Add small margin
    margin = 20
    top = max(0, top - margin)
    bottom = min(img_array.shape[0], bottom + margin)
    left = max(0, left - margin)
    right = min(img_array.shape[1], right + margin)
    
    # Crop to bounding box
    cropped = img_array[top:bottom, left:right]
    
    # Make it square by padding the shorter side
    height, width = cropped.shape
    if height > width:
        diff = height - width
        pad_left = diff // 2
        pad_right = diff - pad_left
        cropped = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    elif width > height:
        diff = width - height
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        cropped = np.pad(cropped, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    
    # Convert back to PIL for resizing
    img_cropped = Image.fromarray(cropped.astype('uint8'))
    
    # Resize to 20x20 (MNIST digits are typically 20x20 centered in 28x28)
    img_resized = img_cropped.resize((20, 20), Image.Resampling.LANCZOS)
    
    # Create 28x28 image with the digit centered (4 pixel padding on each side)
    img_final = Image.new('L', (28, 28), color=0)
    img_final.paste(img_resized, (4, 4))
    
    # Convert to numpy array
    img_array = np.array(img_final, dtype=np.float64)
    
    # Flatten to 784 features (28*28)
    img_flattened = img_array.flatten().reshape(1, -1)
    
    return img_flattened

def create_probability_chart(probabilities):
    """Create a beautiful bar chart for prediction probabilities"""
    colors = ['#ff6b6b', '#ffa502', '#ffd32a', '#7bed9f', '#2ed573', 
              '#1e90ff', '#5352ed', '#a55eea', '#ff6b81', '#ff4757']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(10)),
            y=probabilities,
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='outside',
            textfont=dict(color='white', size=12)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Prediction Probabilities',
            font=dict(color='white', size=18),
            x=0.5
        ),
        xaxis=dict(
            title='Digit',
            tickmode='array',
            tickvals=list(range(10)),
            ticktext=[str(i) for i in range(10)],
            tickfont=dict(color='white', size=14),
            titlefont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            title='Probability',
            tickformat='.0%',
            range=[0, 1.1],
            tickfont=dict(color='white'),
            titlefont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=40, l=40, r=40),
        height=350
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Recognizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Draw a digit (0-9) and watch AI predict in real-time!</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        return
    
    # Initialize session state for canvas key
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Create columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### ✏️ Draw Here")
        
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=True
        )
        
        # Clear button - increment key to reset canvas
        if st.button("🗑️ Clear Canvas", key="clear"):
            st.session_state.canvas_key += 1
            st.rerun()
        
        # Instructions
        st.markdown("""
        <div class="instructions-card">
            <h3>📝 Instructions</h3>
            <ul>
                <li>Draw a single digit (0-9) in the black canvas above</li>
                <li>Use thick strokes for better accuracy</li>
                <li>Center your digit in the canvas</li>
                <li>The prediction updates automatically!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Process and predict
        if canvas_result.image_data is not None:
            # Check if something is drawn
            img_data = canvas_result.image_data
            if np.any(img_data[:, :, :3] > 0):  # Check if any non-black pixels
                # Preprocess the image
                processed_img = preprocess_canvas_image(canvas_result)
                
                if processed_img is not None:
                    # Make prediction
                    prediction = model.predict(processed_img)[0]
                    probabilities = model.predict_proba(processed_img)[0]
                    confidence = probabilities[int(prediction)] * 100
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <p class="label-text">Predicted Digit</p>
                        <p class="big-prediction">{prediction}</p>
                        <p class="confidence-text">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probability chart
                    fig = create_probability_chart(probabilities)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show preprocessed image (debug)
                    with st.expander("🔍 See Preprocessed Image (28x28)"):
                        img_display = processed_img.reshape(28, 28)
                        st.image(img_display, caption="What the model sees", width=150)
            else:
                st.markdown("""
                <div class="prediction-card">
                    <p class="label-text">Waiting for input...</p>
                    <p class="big-prediction" style="color: #4a5568;">?</p>
                    <p style="color: #64748b;">Draw a digit to see prediction</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-card">
                <p class="label-text">Waiting for input...</p>
                <p class="big-prediction" style="color: #4a5568;">?</p>
                <p style="color: #64748b;">Draw a digit to see prediction</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit | Model: Logistic Regression on MNIST Dataset</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
