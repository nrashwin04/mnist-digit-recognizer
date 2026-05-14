import streamlit as st
import torch
from PIL import Image
import tempfile
import plotly.express as px
import pandas as pd
from streamlit_drawable_canvas import st_canvas
import numpy as np
import os
from mnist_cnn import CNN, predict, train

st.set_page_config(layout="wide")

@st.cache_resource
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    if os.path.exists("mnist_cnn.pth"):
        model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))
        return model, device
    return None, device

model, device = get_model()

with st.sidebar:
    st.header("Model Information")
    st.markdown("Architecture Summary:\n- 2 Convolutional Layers\n- BatchNorm2d\n- Dropout (0.25 & 0.5)\n- 2 Fully Connected Layers")
    
    if st.button("View confusion matrix"):
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png")
        else:
            st.error("Confusion matrix not found.")
            
    if st.button("View training curves"):
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png")
        else:
            st.error("Training curves not found.")
            
    st.caption("Model trained on MNIST — 99%+ accuracy")

if model is None:
    st.warning("Model weights (mnist_cnn.pth) not found.")
    if st.button("Train model"):
        with st.spinner("Training model... this will take a few minutes."):
            train()
            st.cache_resource.clear()
            st.rerun()
    st.stop()

col1, col2 = st.columns(2)

with col1:
    tab_draw, tab_upload = st.tabs(["Draw", "Upload"])
    
    with tab_draw:
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = "canvas_0"
            
        stroke_width = st.slider("Stroke width", 10, 30, 18)
        
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_draw = st.button("Predict Canvas", key="btn_pred_draw")
        with col_btn2:
            if st.button("Clear Canvas"):
                st.session_state.canvas_key = "canvas_" + str(np.random.randint(100000))
                st.rerun()
                
    with tab_upload:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, width=280)
            predict_upload = st.button("Predict Upload", key="btn_pred_up")
        else:
            predict_upload = False

run_prediction = False
temp_path = None

if predict_draw and canvas_result.image_data is not None:
    img_data = canvas_result.image_data
    if np.sum(img_data) < 10:
        st.warning("Canvas is blank. Please draw a digit.")
    else:
        r_channel = img_data[:, :, 0].astype(np.uint8)
        pil_img = Image.fromarray(r_channel, mode="L")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp, format="PNG")
            temp_path = tmp.name
        run_prediction = True

elif predict_upload and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name
    run_prediction = True

with col2:
    if run_prediction and temp_path is not None:
        try:
            pred, probs, conf = predict(temp_path, model, device)
            
            st.markdown(f"<h1 style='font-size:96px; text-align:center;'>{pred}</h1>", unsafe_allow_html=True)
            st.metric(label="Confidence", value=f"{conf * 100:.2f}%")
            
            df = pd.DataFrame({
                "Probability": probs,
                "Digit": [str(i) for i in range(10)],
                "Color": ["Predicted" if i == pred else "Other" for i in range(10)]
            })
            
            fig = px.bar(
                df, 
                x="Probability", 
                y="Digit", 
                orientation='h',
                color="Color",
                color_discrete_map={"Predicted": "green", "Other": "gray"}
            )
            fig.update_layout(yaxis={'categoryorder':'category descending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
