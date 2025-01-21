import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import json
import os
from pathlib import Path
import plotly.express as px
from datetime import datetime

# Set TensorFlow options to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set page config at the very beginning
st.set_page_config(
    page_title="RAKSHAK: Empowering Air Defence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define categories and their display names
CATEGORY_MAPPING = {
    '3_long_blades_rotor': 'Long Blade Rotor',
    '3_short_blade_rotor_1': 'Short Blade Rotor Type 1',
    '3_short_blade_rotor_2': 'Short Blade Rotor Type 2',
    'Bird': 'Bird',
    'Bird+mini-helicopter_1': 'Bird with Mini Helicopter Type 1',
    'Bird+mini-helicopter_2': 'Bird with Mini Helicopter Type 2',
    'drone_1': 'Drone Type 1',
    'drone_2': 'Drone Type 2',
    'RC plane_1': 'RC Plane Type 1',
    'RC plane_2': 'RC Plane Type 2'
}

CATEGORIES = list(CATEGORY_MAPPING.keys())

# Custom CSS to remove black columns and improve the dark theme
def local_css():
    st.markdown("""
        <style>
        .stApp {
            background-color: #121212;
            color: #e5e5e5;
        }
        .css-1d391kg {
            background-color: #181818;
            border-radius: 15px;
            padding: 20px;
        }
        .metric-card {
            background-color: #333333;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .confidence-bar {
            background-color: #222222;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            border-left: 4px solid #007bff;
            color: #e5e5e5;
        }
        .alert-box {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .alert-box.warning {
            background-color: #402917;
            border-left: 5px solid #ffc107;
            color: #fcd34d;
        }
        .alert-box.success {
            background-color: #064e3b;
            border-left: 5px solid #10b981;
            color: #6ee7b7;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            background-color: #007bff;
            color: white;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        h1, h2, h3 {
            color: #e5e5e5;
            font-weight: 600;
        }
        .stDataFrame {
            border: none;
            border-radius: 10px;
            overflow: hidden;
        }
        .section-card {
            background-color: #1f1f1f;
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .primary-classification {
            background-color: #333333;
            padding: 15px;
            border-radius: 10px;
            color: #e5e5e5;
        }
        [data-testid="stSidebarNav"] {
            background-color: #1f1f1f;
        }
        .stExpander {
            background-color: #333333;
            border-radius: 10px;
        }
        .stColumns, .stColumn {
            background-color: transparent !important; /* Remove the background from columns */
        }
        </style>
    """, unsafe_allow_html=True)

def load_model_and_mapping():
    try:
        model = load_model("model_outputs/final_classification_model.keras")
        with open("model_outputs/category_mapping.json", "r") as f:
            category_mapping = json.load(f)
        return model, category_mapping
    except Exception as e:
        st.error(f"Error loading model or mapping: {str(e)}")
        return None, None

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def classify_image(image, model):
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_3_idx:
            category = CATEGORIES[idx]
            confidence = predictions[0][idx]
            results.append((category, confidence))
        
        return results
    except Exception as e:
        st.error(f"Classification error: {str(e)}")
        return []

def format_category_name(category):
    return CATEGORY_MAPPING.get(category, category)

def is_invalid_spectrogram(image):
    try:
        image_array = np.array(image)
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        if np.sum(edges) < 5000:  # Simple threshold to check for edge content
            return True

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:  # Faces detected
            return True

        return False
    except Exception as e:
        st.error(f"Error validating image: {str(e)}")
        return True

def create_analytics_section(history_df):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Target Distribution")
        history_df['display_category'] = history_df['category'].map(CATEGORY_MAPPING)
        dist_plot = px.pie(history_df, 
                          names='display_category', 
                          title='Classification Distribution',
                          template='plotly_dark',
                          color_discrete_sequence=px.colors.qualitative.Set3,
                          hole=0.3)
        dist_plot.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_x=0.5,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(dist_plot, use_container_width=True)
    
    with col2:
        st.subheader("📈 Confidence Trends")
        trend_plot = px.line(history_df, 
                            x='timestamp', 
                            y='confidence',
                            color='display_category',
                            title='Confidence Trends by Category',
                            template='plotly_dark')
        trend_plot.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Time",
            yaxis_title="Confidence (%)",
            title_x=0.5,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(trend_plot, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def get_dashboard_metrics(history):
    if not history:
        return 0, 0, {}
    
    today = datetime.now().date()
    today_scans = sum(1 for x in history if x['timestamp'].date() == today)
    alerts = sum(1 for x in history 
                if any(keyword in x['category'].lower() 
                      for keyword in ['drone', 'rotor', 'rc plane']))
    
    category_counts = {}
    for entry in history:
        display_category = CATEGORY_MAPPING[entry['category']]
        category_counts[display_category] = category_counts.get(display_category, 0) + 1
    
    return today_scans, alerts, category_counts

def main():
    local_css()
    
    # Initialize session state
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []

    # Load model
    model, category_mapping = load_model_and_mapping()

    # Sidebar
    with st.sidebar:
        logo_path = Path(r"C:\\Users\\sanvi\\OneDrive\\Desktop\\logo_enhanced.png")
        if logo_path.exists():
            try:
                st.image(str(logo_path), use_container_width=True)
            except Exception as e:
                st.warning("Could not load logo image")
        else:
            st.title("RAKSHAK")
        
        # Initialize page selection
        page = st.radio("Navigation", ["Dashboard", "Micro-Doppler Classification", "About Us"])

    if page == "Dashboard":
        st.title("🎯 RAKSHAK Dashboard")

        # Get metrics for Dashboard
        total_scans, total_alerts, category_counts = get_dashboard_metrics(st.session_state.classification_history)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Scans Today", total_scans, f"+{total_scans}")
        with col2:
            st.metric("🚨 Alerts Generated", total_alerts, f"+{total_alerts}")
        with col3:
            status = "Active" if model is not None else "Model Loading Error"
            st.metric("⚡ System Status", status, "100%" if status == "Active" else "Error")
        
        # Analytics section
        if st.session_state.classification_history:
            history_df = pd.DataFrame(st.session_state.classification_history)
            create_analytics_section(history_df)
            
            # Recent classifications table
            st.subheader("Recent Classifications")
            display_df = history_df.copy()
            display_df['category'] = display_df['category'].map(CATEGORY_MAPPING)
            display_df = display_df[['timestamp', 'category', 'confidence']].sort_values('timestamp', ascending=False).head(10)
            display_df.columns = ['Timestamp', 'Target Type', 'Confidence (%)']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No classification history available yet.")

    elif page == "Micro-Doppler Classification":
        st.title("🔍 Micro-Doppler Signature Classification")

        with st.expander("ℹ️ About Micro-Doppler Signature Classification"):
            st.markdown("""
            Micro-Doppler signatures are unique patterns created by the micro-motions of targets, such as the rotation 
            of propellers or the flapping of bird wings. These signatures help in identifying and classifying aerial targets 
            with high precision.
            
            **Supported Format**: JPEG images of micro-Doppler spectrograms
            """)

        
        uploaded_image = st.file_uploader("📤 Upload Micro-Doppler Signature", type=["jpg", "jpeg"])
        
        if uploaded_image:
            try:
                image = Image.open(uploaded_image)
                if is_invalid_spectrogram(image):
                    st.error("Error: The uploaded image is not recognized as a valid spectrogram.")
                    return
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### 🖼️ Input Spectrogram")
                    st.image(image, caption="Uploaded Micro-Doppler Signature", use_container_width=True)
                with col2:
                    st.markdown("#### 📊 Analysis Results")
                    with st.spinner("Analyzing micro-Doppler signature..."):
                        results = classify_image(image, model)
                    
                    if results:
                        top_category, top_confidence = results[0]
                        confidence_percentage = top_confidence * 100
                        
                        st.markdown("### 🎯 Primary Classification")
                        st.markdown(f"""
                        <div class='primary-classification'>
                            <h4>Target Type: {format_category_name(top_category)}</h4>
                            <h4>Confidence: {confidence_percentage:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🔄 Alternative Possibilities")
                        for category, confidence in results[1:]:
                            conf_percentage = confidence * 100
                            st.markdown(
                                f"""<div class="confidence-bar">
                                    {format_category_name(category)}: {conf_percentage:.2f}%
                                    </div>""",
                                unsafe_allow_html=True
                            )

                        st.markdown("#### ⚠️ Threat Assessment")
                        if any(keyword in top_category.lower() for keyword in ['drone', 'rotor', 'rc plane']):
                            st.markdown(
                                """<div class="alert-box warning">
                                    🚨 <strong>ALERT:</strong> Potential unauthorized aircraft detected!
                                    <br><br>
                                    <strong>Recommended Actions:</strong>
                                    <ol>
                                        <li>📍 Monitor target trajectory</li>
                                        <li>📢 Alert nearest control center</li>
                                        <li>🛡️ Prepare countermeasures if necessary</li>
                                    </ol>
                                    </div>""",
                                unsafe_allow_html=True
                            )
                        elif 'bird' in top_category.lower():
                            st.markdown(
                                """<div class="alert-box success">
                                    ✅ <strong>Safe:</strong> Biological target identified
                                    </div>""",
                                unsafe_allow_html=True
                            )


                        st.session_state.classification_history.append({
                            'timestamp': datetime.now(),
                            'category': top_category,
                            'confidence': confidence_percentage
                        })

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif page == "About Us":
        st.title("ℹ️ About RAKSHAK")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Our Mission
            RAKSHAK is at the forefront of air defence technology, combining cutting-edge machine learning with robust security protocols to protect airspaces effectively.
            
            ### Key Features
            - 🎯 **Real-time Classification**: Instant identification of aerial targets through micro-Doppler signatures
            - 🤖 **Advanced AI**: State-of-the-art machine learning algorithms for precise target classification
            - 📊 **Detailed Analytics**: Comprehensive analysis and reporting of classification results
            - 🔄 **Continuous Learning**: Self-improving classification accuracy through data analysis
            """)
            st.subheader("Technology Stack")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🧠 Machine Learning")
            st.markdown("- TensorFlow\n- Deep Neural Networks\n- Signal Processing")
        with col2:
            st.markdown("### 🔒 Security")
            st.markdown("- Real-time Monitoring\n- Alert System\n- Threat Assessment")
        with col3:
            st.markdown("### 📈 Analytics")
            st.markdown("- Performance Metrics\n- Historical Analysis\n- Trend Detection")

if __name__ == "__main__":
    main()
