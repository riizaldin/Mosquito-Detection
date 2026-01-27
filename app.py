import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Mosquito Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center;
        color: #3D9DF3 !important;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.3rem !important;
        text-align: center;
        color: #4A5568 !important;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for mosquito species
CLASS_NAMES = ['Aedes aegypti', 'Aedes albopictus', 'Culex quinquefasciatus']

# Model paths
MODEL_PATHS = {
    'swin': 'best_swin_t.pth',
    'efficientnet': 'best_efficientnetv2_s.pth',
    'convnext': 'best_convnext_t.pth'
}

# Ensemble weights
ENSEMBLE_WEIGHTS = [0.35, 0.35, 0.30]  # Swin, EfficientNet, ConvNeXt

@st.cache_resource
def load_models():
    """Load all three models for ensemble prediction"""
    num_classes = len(CLASS_NAMES)
    models_dict = {}
    
    with st.spinner("Loading models... Please wait..."):
        # Load Swin Transformer Tiny
        swin_model = models.swin_t(weights=None)
        swin_model.head = nn.Sequential(
            nn.Linear(swin_model.head.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        if os.path.exists(MODEL_PATHS['swin']):
            swin_model.load_state_dict(torch.load(MODEL_PATHS['swin'], map_location=DEVICE))
            swin_model = swin_model.to(DEVICE)
            swin_model.eval()
            models_dict['swin'] = swin_model
        
        # Load EfficientNetV2-S
        efficientnet_model = models.efficientnet_v2_s(weights=None)
        efficientnet_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        if os.path.exists(MODEL_PATHS['efficientnet']):
            efficientnet_model.load_state_dict(torch.load(MODEL_PATHS['efficientnet'], map_location=DEVICE))
            efficientnet_model = efficientnet_model.to(DEVICE)
            efficientnet_model.eval()
            models_dict['efficientnet'] = efficientnet_model
        
        # Load ConvNeXt Tiny
        convnext_model = models.convnext_tiny(weights=None)
        convnext_model.classifier[2] = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        if os.path.exists(MODEL_PATHS['convnext']):
            convnext_model.load_state_dict(torch.load(MODEL_PATHS['convnext'], map_location=DEVICE))
            convnext_model = convnext_model.to(DEVICE)
            convnext_model.eval()
            models_dict['convnext'] = convnext_model
    
    return models_dict

def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_tta_transforms():
    """Get Test Time Augmentation transforms"""
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Rotation 10 degrees
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Contrast adjustment
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]
    return tta_transforms

def predict_ensemble(image, models_dict, use_tta=False):
    """Make ensemble prediction on an image"""
    if not models_dict:
        st.error("No models loaded. Please check if model files exist.")
        return None, None
    
    # Prepare image
    if use_tta:
        transforms_list = get_tta_transforms()
    else:
        transforms_list = [get_transform()]
    
    ensemble_predictions = []
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            tta_preds = []
            
            for transform in transforms_list:
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                tta_preds.append(probs)
            
            # Average TTA predictions
            avg_pred = np.mean(tta_preds, axis=0)
            ensemble_predictions.append(avg_pred)
    
    # Weighted ensemble
    model_list = ['swin', 'efficientnet', 'convnext']
    final_probs = np.zeros(len(CLASS_NAMES))
    for i, model_name in enumerate(model_list):
        if model_name in models_dict:
            weight = ENSEMBLE_WEIGHTS[i]
            final_probs += ensemble_predictions[i] * weight
    
    # Normalize
    final_probs = final_probs / final_probs.sum()
    
    predicted_class = np.argmax(final_probs)
    confidence = final_probs[predicted_class]
    
    return predicted_class, final_probs

def get_species_info(species_name):
    """Get information about mosquito species"""
    info = {
        'Aedes aegypti': {
            'description': 'Also known as the yellow fever mosquito, is a vector for several tropical diseases including dengue fever, chikungunya, Zika fever, Mayaro and yellow fever.',
            'characteristics': '• Black with white markings\n• Lyre-shaped pattern on thorax\n• Active during daytime',
            'danger_level': 'High',
            'color': '#E53935'
        },
        'Aedes albopictus': {
            'description': 'Also known as the Asian tiger mosquito, is a vector for many viral pathogens, including yellow fever, dengue fever, and Chikungunya fever.',
            'characteristics': '• Black with distinctive white stripes\n• Single white stripe on thorax\n• Active during day and early evening',
            'danger_level': 'High',
            'color': '#FB8C00'
        },
        'Culex quinquefasciatus': {
            'description': 'Also known as the southern house mosquito, is a vector of Wuchereria bancrofti, avian malaria, and arboviruses including St. Louis encephalitis virus, Western equine encephalitis virus, Zika virus and West Nile virus.',
            'characteristics': '• Brown or tan color\n• No distinctive markings\n• Active at night',
            'danger_level': 'Medium',
            'color': '#FDD835'
        }
    }
    return info.get(species_name, {})

def main():
    # Header
    st.markdown('<p class="main-header">Mosquito Species Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Information")
    st.sidebar.markdown("---")
    
    # Model info
    st.sidebar.subheader("Model Information")
    st.sidebar.info("""
    **Ensemble Architecture:**
    - Swin Transformer Tiny (35%)
    - EfficientNetV2-S (35%)
    - ConvNeXt Tiny (30%)
    
    **Classes:**
    - Aedes aegypti
    - Aedes albopictus
    - Culex quinquefasciatus
    """)
    
    # TTA option
    use_tta = st.sidebar.checkbox("Enable Test Time Augmentation (TTA)", value=True, 
                                   help="TTA applies multiple transformations to improve accuracy but takes longer")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    # Load models
    try:
        models_dict = load_models()
        if not models_dict:
            st.error("No model files found. Please ensure the model .pth files are in the same directory as this app.")
            return
        st.sidebar.success(f"{len(models_dict)} models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a mosquito image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("Detect Mosquito Species", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, probabilities = predict_ensemble(image, models_dict, use_tta)
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state['prediction'] = CLASS_NAMES[predicted_class]
                        st.session_state['probabilities'] = probabilities
    
    with col2:
        st.subheader("Results")
        
        if 'prediction' in st.session_state:
            predicted_species = st.session_state['prediction']
            probabilities = st.session_state['probabilities']
            confidence = probabilities[CLASS_NAMES.index(predicted_species)]
            
            # Prediction result
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #FFFFFF; margin: 0; font-weight: 500; font-size: 1.3rem;">Predicted Species</h2>
                    <h1 style="color: #FFFFFF; margin: 15px 0; font-weight: 600; font-size: 2rem;">{predicted_species}</h1>
                    <h3 style="color: #ECF0F1; margin: 0; font-weight: 400; font-size: 1.1rem;">Confidence: {confidence*100:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence bars for all classes
            st.markdown("### Confidence Levels")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = probabilities[i]
                st.write(f"**{class_name}**")
                st.progress(float(prob))
                st.caption(f"{prob*100:.2f}%")
            
            # Species information
            st.markdown("---")
            st.markdown("### Species Information")
            species_info = get_species_info(predicted_species)
            
            if species_info:
                st.markdown(f"**Danger Level:** <span style='color: {species_info['color']}; font-weight: bold;'>{species_info['danger_level']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {species_info['description']}")
                st.markdown("**Key Characteristics:**")
                st.text(species_info['characteristics'])
        else:
            st.info("Upload an image and click 'Detect Mosquito Species' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Developed using Streamlit and PyTorch</p>
            <p>Ensemble Model: Swin Transformer + EfficientNetV2 + ConvNeXt</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
