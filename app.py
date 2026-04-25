import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm



st.set_page_config(
    page_title="Flood Area Detection",
    page_icon=" ",
    layout="wide"
)

# Loss / metric definitions, matching training exactly
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.reshape(y_true, [-1])
    y_pred_f     = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def tversky_loss(y_true, y_pred, alpha=0.7, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    return 1.0 - (tp + smooth) / (tp + alpha*fn + (1-alpha)*fp + smooth)

def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tversky_loss(y_true, y_pred, alpha=0.7)

def iou_metric(y_true, y_pred):
    y_pred_bin   = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f     = tf.reshape(y_true,     [-1])
    y_pred_f     = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_coef(y_true, y_pred):
    y_pred_bin   = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f     = tf.reshape(y_true,     [-1])
    y_pred_f     = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6
    )

# Preprocessing, matching training exactly
BACKBONE         = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
IMG_SIZE         = 256

def normalize_sar(arr):
    arr = arr.astype(np.float32)
    p2  = np.percentile(arr, 2)
    p98 = np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    return (arr - p2) / (p98 - p2 + 1e-8)

def preprocess_image(uploaded_file):
    """
    Takes an uploaded file, returns a (1, 256, 256, 3) array
    ready to feed into the model, plus a display-friendly RGB image.
    """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, None

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # Simulate VH + VV from grayscale (matches training workaround)
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray_norm = normalize_sar(gray_res)

    # 2-channel → 3-channel (duplicate VH as 3rd channel, same as Cell 4)
    img_2ch = np.stack([gray_norm, gray_norm], axis=-1)          # (256,256,2)
    img_3ch = np.concatenate([img_2ch,
                               img_2ch[:, :, 0:1]], axis=-1)     # (256,256,3)

    # Apply ResNet34 ImageNet preprocessing (same as Cell 6)
    img_3ch = preprocess_input(img_3ch)

    return img_3ch[np.newaxis, ...], display   # (1,256,256,3), (256,256,3)

# Model loading
@st.cache_resource
def load_flood_model():
    return tf.keras.models.load_model(
        "final_flood_model.keras",
        custom_objects={
            "combined_loss": combined_loss,
            "iou_metric":    iou_metric,
            "dice_coef":     dice_coef,
        },
        compile=False
    )

model = load_flood_model()

#  UI
st.title(" Flood Area Detection")
st.markdown(
    "Upload a satellite image to detect flood-affected regions using a "
    "**ResNet34 U-Net** model trained on SAR imagery."
)
st.markdown("---")

st.sidebar.header(" Settings")
threshold = st.sidebar.slider(
    "Prediction Threshold", min_value=0.1, max_value=0.9,
    value=0.5, step=0.05,
    help="Pixels above this probability are classified as flooded."
)
show_raw_prob = st.sidebar.checkbox("Show raw probability map", value=False)

uploaded_file = st.file_uploader(
    "Choose a satellite image...",
    type=["jpg", "png", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:

    img_input, display = preprocess_image(uploaded_file)

    if img_input is None:
        st.error("Could not read the uploaded file. Please upload a valid image.")
        st.stop()

    with st.spinner("Running flood detection..."):
        pred      = model.predict(img_input, verbose=0)[0]   # (256,256,1)
        pred_sq   = pred.squeeze()                            # (256,256)
        pred_mask = (pred_sq > threshold).astype(np.uint8)

    flood_pct = pred_mask.mean() * 100

    st.markdown("  Results")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Flood Coverage",  f"{flood_pct:.2f}%")
    col_b.metric("Threshold Used",  f"{threshold}")
    col_c.metric("Max Flood Prob",  f"{pred_sq.max():.3f}")
    st.markdown(" ")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(display, caption=" Original Image",
                 use_container_width=True)

    with col2:
        if show_raw_prob:
            prob_disp = (pred_sq * 255).astype(np.uint8)
            st.image(prob_disp, caption=" Probability Map",
                     use_container_width=True, clamp=True)
        else:
            mask_disp = (pred_mask * 255).astype(np.uint8)
            st.image(mask_disp, caption=" Predicted Flood Mask",
                     use_container_width=True, clamp=True)

    with col3:
        overlay = display.copy()
        overlay[pred_mask == 1] = [220, 50, 50]
        blended = cv2.addWeighted(display, 0.45, overlay, 0.55, 0)
        st.image(blended, caption=" Flood Overlay",
                 use_container_width=True)

    st.markdown(" ")

    if flood_pct < 5:
        st.success(f" Low flood risk detected — {flood_pct:.2f}% of area")
    elif flood_pct < 25:
        st.warning(f" Moderate flooding detected — {flood_pct:.2f}% of area")
    else:
        st.error(f" Severe flooding detected — {flood_pct:.2f}% of area")

    st.caption(
        " Note: This model was trained on 2-channel SAR (VH/VV) data. "
        "RGB images are converted to grayscale to simulate SAR input. "
        "For accurate results, provide real SAR satellite imagery."
    )

else:
    st.info(" Upload a satellite image above to get started.")
    st.markdown("""
    How it works:
    1. Upload a satellite image (JPG / PNG / GeoTIFF)
    2. The model classifies each pixel as flooded or non-flooded
    3. Results shown as a segmentation mask and colour overlay

    Model: ResNet34 U-Net with ImageNet pretrained encoder
    Input: 2-channel SAR imagery (VH + VV polarisation)
    Metrics: IoU ≈ 0.59 | Dice ≈ 0.70 on held-out test images
    """)