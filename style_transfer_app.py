import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO

# Hide Streamlit menu and footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1em;
            font-size: 16px;
            border-radius: 8px;
        }
        .stDownloadButton>button {
            background-color: #2196F3;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.title("🎨 Neural Style Transfer")
st.markdown(
    "Transform your photos into artwork by combining the **content of one image** with the **style of another**. "
    "Powered by TensorFlow and Magenta's pre-trained model."
)

# Load model
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

model = load_model()

def load_image(uploaded_file, max_dim=512):
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image).astype(np.float32) / 255.0

    img = tf.convert_to_tensor(img)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = tf.reduce_max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor[0]  # remove batch dimension
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(tensor)

# Upload Section
st.header("📁 Upload Images")
col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    if content_file:
        st.image(content_file, caption="🖼️ Content Image", use_column_width=True)

with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    if style_file:
        st.image(style_file, caption="🎨 Style Image", use_column_width=True)

# Stylize button
if content_file and style_file:
    stylize = st.button("✨ Stylize Image")

    if stylize:
        with st.spinner("Applying style... Please wait."):
            content_image = load_image(content_file)
            style_image = load_image(style_file)

            stylized_image = model(content_image, style_image)[0]
            result = tensor_to_image(stylized_image)

            st.success("✅ Stylization Complete!")
            st.image(result, caption="🖌️ Stylized Output", use_column_width=True)

            # Convert to bytes for download
            buf = BytesIO()
            result.save(buf, format="PNG")
            buf.seek(0)

            st.download_button("⬇️ Download Stylized Image", buf, file_name="stylized_output.png", mime="image/png")

else:
    st.info("👆 Please upload both content and style images to begin.")

# Footer
st.markdown("""
<hr>
<p style='text-align: center'>
    Made with ❤️ using <b>TensorFlow</b> & <b>Streamlit</b><br>
    <small>© 2025 NeuralArtify Inc.</small>
</p>
""", unsafe_allow_html=True)
