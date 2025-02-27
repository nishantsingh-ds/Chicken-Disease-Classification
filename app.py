import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import io

# Must be the first Streamlit command
st.set_page_config(page_title="Chicken Disease Classifier", layout="wide")

# -------------------------------
# 1. Caching / Resource Loading
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "artifacts/prepare_base_model/base_model_updated.h5"  # Adjust if needed
    model = tf.keras.models.load_model(model_path)
    return model

def predict_image(model, image_array, class_names):
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index]
    predicted_class = class_names[predicted_index]
    return predicted_class, confidence

# --------------------------------
# 2. Streamlit App Layout
# --------------------------------
def main():
    custom_css()  # <-- Keep your custom styling function here
    
    # ---------- SIDEBAR ----------
    st.sidebar.title("Navigation")
    st.sidebar.markdown("### Instructions:")
    st.sidebar.write(
        """
        1. Upload an image of a chicken on the **main page**.
        2. Click **Classify Image** to run the model.
        3. View the predicted disease and confidence score.
        4. Switch tabs to see Confidence Chart or Debug Logs.
        """
    )
    st.sidebar.write("---")
    st.sidebar.markdown(
        "[GitHub Repo](https://github.com/nishantsingh-ds/Chicken-Disease-Classification)"
    )

    # ---------- MAIN TITLE ----------
    st.markdown("<h1 style='text-align: center;'>🐔 Chicken Disease Classifier 🐔</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align: center; font-size: 18px;'>
        Upload an image of a chicken, and this model will classify potential diseases.<br>
        This demo showcases a <strong>Convolutional Neural Network (CNN)</strong> or other deep learning architecture 
        trained specifically for chicken disease detection.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # ---------- FILE UPLOADER ----------
    uploaded_file = st.file_uploader(
        "Choose an image of a chicken...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a JPEG or PNG image of a chicken to classify."
    )

    # Example: Classes your model can predict
    class_names = ["Healthy", "Infected"]  # Replace with your actual class labels
    
    # Display the uploaded image
    if uploaded_file is not None:
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.subheader("Uploaded Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.subheader("Classification")
            st.write("Click the button below to classify the image.")
            
            if st.button("Classify Image", use_container_width=True):
                with st.spinner("Loading model & preparing for classification..."):
                    image_resized = image.resize((224, 224))
                    image_array = np.array(image_resized) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    model = load_model()  # Using @st.cache_resource

                    # Simulate a short progress bar for user experience
                    progress_bar = st.progress(0)
                    for i in range(1, 101):
                        time.sleep(0.01)
                        progress_bar.progress(i)

                    predicted_class, confidence = predict_image(model, image_array, class_names)
                    st.success("Classification Completed!")
                    st.balloons()
                    
                st.write(f"**Predicted Class:** :mag_right: `{predicted_class}`")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                
                # ---------- TABS FOR ADDITIONAL DETAILS ----------
                tab1, tab2 = st.tabs(["Confidence Chart", "Debug Logs"])
                
                with tab1:
                    st.markdown("#### Prediction Confidence across Classes")
                    st.bar_chart(prediction_chart_data(model, image_array, class_names))
                
                with tab2:
                    st.markdown("#### Debug Logs & Info")
                    st.write("Here you might display additional logs, raw prediction values, or model layers.")
    else:
        st.info("Please upload an image to begin.")
        

def prediction_chart_data(model, image_array, class_names):
    import pandas as pd
    preds = model.predict(image_array)[0]
    data = pd.DataFrame({
        "Class": class_names,
        "Confidence": preds
    })
    return data.set_index("Class")

def custom_css():
    st.markdown(
        """
        <style>
        h2 {
            text-align: center !important;
        }
        .css-1cpxqw2, .css-keje6w {
            color: #333 !important;
        }
        .appview-container {
            background-color: #FAFAFA;
        }
        [data-testid="stSidebar"] {
            background-color: #FFF8E1;
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #FFCA28 , #FFEE58);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
