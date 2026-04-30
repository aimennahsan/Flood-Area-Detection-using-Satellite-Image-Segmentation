import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm

st.set_page_config(
    page_title="Flood Area Detection",
    page_icon=" ",
    layout="wide"
)

# Loss / metric definitions 
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

# Preprocessing 
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
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, None
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    display   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_res  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray_norm = normalize_sar(gray_res)
    img_2ch   = np.stack([gray_norm, gray_norm], axis=-1)
    img_3ch   = np.concatenate([img_2ch, img_2ch[:, :, 0:1]], axis=-1)
    img_3ch   = preprocess_input(img_3ch)
    return img_3ch[np.newaxis, ...], display

#  Model loading 
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

# Shared matplotlib style 
def style_ax(ax):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.grid(True, alpha=0.15, color='white')

# Navigation 
page = st.sidebar.selectbox(
    "Navigation",
    ["Flood Detection", "Image Analysis"]
)

# PAGE 1 — Flood Detection
if page == "Flood Detection":

    st.title("Flood Area Detection")
    st.markdown(
        "Upload a satellite image to detect flood-affected regions using a "
        "**ResNet34 U-Net** model trained on SAR imagery."
    )
    st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.header("Settings")
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
            st.error("Could not read the uploaded file.")
            st.stop()

        with st.spinner("Running flood detection..."):
            pred      = model.predict(img_input, verbose=0)[0]
            pred_sq   = pred.squeeze()
            pred_mask = (pred_sq > threshold).astype(np.uint8)

        st.session_state['pred_sq']   = pred_sq
        st.session_state['pred_mask'] = pred_mask
        st.session_state['display']   = display
        st.session_state['threshold'] = threshold
        st.session_state['filename']  = uploaded_file.name

        flood_pct = pred_mask.mean() * 100

        st.markdown("### Results")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Flood Coverage", f"{flood_pct:.2f}%")
        col_b.metric("Threshold Used", f"{threshold}")
        col_c.metric("Max Flood Prob", f"{pred_sq.max():.3f}")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(display, caption="Original Image",
                     use_container_width=True)

        with col2:
            if show_raw_prob:
                prob_raw = pred_sq.copy()
                max_val  = prob_raw.max()
                if max_val > 0:
                    prob_stretched = (prob_raw / max_val * 255).astype(np.uint8)
                else:
                    prob_stretched = np.zeros_like(prob_raw, dtype=np.uint8)
                prob_colored = cv2.applyColorMap(prob_stretched, cv2.COLORMAP_JET)
                prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
                st.image(prob_colored,
                         caption=f"Probability Map (max={pred_sq.max():.3f})",
                         use_container_width=True)
                if max_val < 0.1:
                    st.caption(
                        f"Max probability is only {max_val:.3f} — "
                        "image stretched for visibility. "
                        "Model is confident this area is NOT flooded."
                    )
            else:
                mask_disp = (pred_mask * 255).astype(np.uint8)
                st.image(mask_disp, caption="Predicted Flood Mask",
                         use_container_width=True, clamp=True)

        with col3:
            overlay = display.copy()
            overlay[pred_mask == 1] = [220, 50, 50]
            blended = cv2.addWeighted(display, 0.45, overlay, 0.55, 0)
            st.image(blended, caption="Flood Overlay",
                     use_container_width=True)

        st.markdown("---")

        if flood_pct < 5:
            st.success(f"Low flood risk detected — {flood_pct:.2f}% of area")
        elif flood_pct < 25:
            st.warning(f"Moderate flooding detected — {flood_pct:.2f}% of area")
        else:
            st.error(f"Severe flooding detected — {flood_pct:.2f}% of area")

        st.caption(
            "Note: This model was trained on 2-channel SAR (VH/VV) data. "
            "RGB images are converted to grayscale to simulate SAR input. "
            "For accurate results, provide real SAR satellite imagery."
        )

        st.info("Go to Image Analysis in the sidebar to see detailed statistics for this image.")

    else:
        st.info("Upload a satellite image above to get started.")
        st.markdown("""
        **How it works:**
        1. Upload a satellite image (JPG / PNG / GeoTIFF)
        2. The model classifies each pixel as **flooded** or **non-flooded**
        3. Results shown as a segmentation mask and colour overlay

        **Model:** ResNet34 U-Net with ImageNet pretrained encoder
        **Input:** 2-channel SAR imagery (VH + VV polarisation)
        **Metrics:** IoU 0.66 | Dice 0.78 on held-out test images
        """)

# PAGE 2 — Image Analysis
elif page == "Image Analysis":

    st.title("Image Analysis")
    st.markdown("Detailed statistics and analysis for the last uploaded image.")
    st.markdown("---")

    if 'pred_sq' not in st.session_state:
        st.warning("No image analysed yet. Go to Flood Detection and upload an image first.")
        st.stop()

    pred_sq   = st.session_state['pred_sq']
    pred_mask = st.session_state['pred_mask']
    display   = st.session_state['display']
    threshold = st.session_state['threshold']
    filename  = st.session_state['filename']

    st.markdown(f"**Analysing:** {filename}")
    st.markdown("---")

    #  Section 1: Per-image metrics 
    st.header("Image Metrics")

    flood_pct       = pred_mask.mean() * 100
    nonflood_pct    = 100 - flood_pct
    flood_pixels    = int(pred_mask.sum())
    nonflood_pixels = int((pred_mask == 0).sum())
    total_pixels    = flood_pixels + nonflood_pixels
    max_prob        = float(pred_sq.max())
    mean_prob       = float(pred_sq.mean())
    high_conf_pct   = float((pred_sq > 0.7).mean() * 100)
    low_conf_pct    = float(((pred_sq > 0.3) & (pred_sq < 0.7)).mean() * 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Flood Coverage",   f"{flood_pct:.2f}%")
    col2.metric("Max Flood Prob",   f"{max_prob:.3f}")
    col3.metric("Mean Flood Prob",  f"{mean_prob:.3f}")
    col4.metric("High Conf Pixels", f"{high_conf_pct:.1f}%",
                help="Pixels where model is more than 70% confident they are flooded")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Flooded Pixels",     f"{flood_pixels:,}")
    col6.metric("Non-Flooded Pixels", f"{nonflood_pixels:,}")
    col7.metric("Total Pixels",       f"{total_pixels:,}")
    col8.metric("Uncertain Pixels",   f"{low_conf_pct:.1f}%",
                help="Pixels where model probability is between 0.3 and 0.7")

    st.markdown("---")

    #  Section 2: Probability distribution 
    st.header("Pixel Probability Distribution")
    st.markdown(
        "Shows how confident the model is across all pixels. "
        "Peaks near 0 mean the model is confident those pixels are not flooded. "
        "Peaks near 1 mean the model is confident those pixels are flooded. "
        "Pixels in the middle are uncertain."
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor('#0e1117')
    style_ax(ax1)
    ax1.hist(pred_sq.flatten(), bins=80, color='steelblue',
             edgecolor='none', alpha=0.85)
    ax1.axvline(x=threshold, color='tomato', linestyle='--',
                linewidth=2, label=f'Threshold ({threshold})')
    ax1.axvline(x=0.7, color='orange', linestyle=':',
                linewidth=1.5, label='High confidence boundary (0.7)')
    ax1.set_xlabel("Flood Probability")
    ax1.set_ylabel("Number of Pixels")
    ax1.set_title("Distribution of Flood Probabilities Across All Pixels")
    ax1.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown("---")

    #  Section 3: Pixel breakdown 
    st.header("Pixel Classification Breakdown")

    col_left, col_right = st.columns(2)

    with col_left:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        wedges, texts, autotexts = ax2.pie(
            [nonflood_pixels, flood_pixels],
            labels=["Non-Flooded", "Flooded"],
            colors=["royalblue", "tomato"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={'color': 'white', 'fontsize': 12}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(12)
        ax2.set_title("Flooded vs Non-Flooded Pixels",
                      color='white', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_right:
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        fig3.patch.set_facecolor('#0e1117')
        style_ax(ax3)
        confidence_bins = [
            int((pred_sq < 0.3).sum()),
            int(((pred_sq >= 0.3) & (pred_sq < 0.7)).sum()),
            int((pred_sq >= 0.7).sum())
        ]
        bar_labels = ["Confident\nNon-Flood\n(< 0.3)",
                      "Uncertain\n(0.3 - 0.7)",
                      "Confident\nFlood\n(> 0.7)"]
        bar_colors = ["royalblue", "gold", "tomato"]
        bars = ax3.bar(bar_labels, confidence_bins, color=bar_colors)
        ax3.set_title("Pixel Confidence Breakdown", color='white', fontsize=13)
        ax3.set_ylabel("Number of Pixels")
        for bar, val in zip(bars, confidence_bins):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 200,
                     f'{val:,}',
                     ha='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.markdown("---")

    #  Section 4: Threshold sensitivity 
    st.header("Threshold Sensitivity")
    st.markdown(
        "Shows how flood coverage percentage changes as you move the threshold. "
        "This helps you understand how sensitive the prediction is to your threshold choice."
    )

    thresholds      = np.arange(0.1, 0.95, 0.05)
    flood_at_thresh = [(pred_sq > t).mean() * 100 for t in thresholds]

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fig4.patch.set_facecolor('#0e1117')
    style_ax(ax4)
    ax4.plot(thresholds, flood_at_thresh, color='steelblue',
             linewidth=2.5, marker='o', markersize=5)
    ax4.axvline(x=threshold, color='tomato', linestyle='--',
                linewidth=2, label=f'Current threshold ({threshold})')
    ax4.axhline(y=flood_pct, color='orange', linestyle=':',
                linewidth=1.5, label=f'Current flood % ({flood_pct:.1f}%)')
    ax4.set_xlabel("Threshold Value")
    ax4.set_ylabel("Flood Coverage (%)")
    ax4.set_title("Flood Coverage vs Prediction Threshold")
    ax4.legend(facecolor='#1a1a2e', labelcolor='white')
    ax4.set_xlim(0.1, 0.9)
    ax4.set_ylim(0, 100)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown("---")

    # Section 5: Summary table
    st.header("Summary Table")

    metrics = [
        ("Flood Coverage",                      f"{flood_pct:.2f}%"),
        ("Non-Flood Coverage",                  f"{nonflood_pct:.2f}%"),
        ("Flooded Pixels",                      f"{flood_pixels:,}"),
        ("Non-Flooded Pixels",                  f"{nonflood_pixels:,}"),
        ("Total Pixels",                        f"{total_pixels:,}"),
        ("Max Flood Probability",               f"{max_prob:.4f}"),
        ("Mean Flood Probability",              f"{mean_prob:.4f}"),
        ("High Confidence Flood Pixels (>0.7)", f"{high_conf_pct:.2f}%"),
        ("Uncertain Pixels (0.3 - 0.7)",        f"{low_conf_pct:.2f}%"),
        ("Threshold Used",                      f"{threshold}"),
    ]

    rows = "".join(f"| {m} | {v} |\n" for m, v in metrics)
    st.markdown(f"| Metric | Value |\n|--------|-------|\n{rows}")

    st.markdown("---")
    st.caption(
        "Note: These metrics are computed directly from the model output for "
        "this specific uploaded image. No ground truth mask is available at "
        "inference time so Dice and IoU cannot be computed here."
    )