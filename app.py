"""
NeuroScan-AI: Diagnostic Demonstration Web Application
Interactive Streamlit interface for automated Brain Tumor MRI classification
using Deep Transfer Learning (MobileNetV2 backbone).
"""

import os
import sys
from PIL import Image
import streamlit as st
import pandas as pd
import altair as alt

# Ensure src modules are resolvable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.dataset import CLASS_NAMES, CLASS_DISPLAY_NAMES
from src.predict import NeuroScanPredictor, SEVERITY_MAPPINGS

st.set_page_config(
    page_title="NeuroScan-AI | Brain Tumor Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1a365d;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Initializes and caches the Deep Learning model predictor."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "neuroscan_best_model.pt")
    if not os.path.exists(model_path):
        return None
    try:
        return NeuroScanPredictor(model_path=model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    predictor = load_predictor()

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=70)
    st.sidebar.title("NeuroScan-AI")
    st.sidebar.markdown("**Deep Learning Brain MRI Diagnostic System**")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Clinical Project Details")
    st.sidebar.markdown("""
    - **Architecture**: MobileNetV2 Transfer Learning
    - **Task**: 4-Class Brain MRI Classification
    - **Classes**: Glioma, Meningioma, Pituitary, Normal
    - **Optimization**: Cross-Entropy + AdamW + Cosine LR
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Test Samples")
    sample_choice = st.sidebar.selectbox(
        "Select an authentic benchmark sample:",
        ["None (Upload custom MRI)", "Sample 1: Glioma Tumor", "Sample 2: Meningioma Tumor", "Sample 3: Normal Brain Scan", "Sample 4: Pituitary Tumor"]
    )

    sample_mapping = {
        "Sample 1: Glioma Tumor": ("data/test/glioma", "Glioma"),
        "Sample 2: Meningioma Tumor": ("data/test/meningioma", "Meningioma"),
        "Sample 3: Normal Brain Scan": ("data/test/no_tumor", "Normal (No Tumor)"),
        "Sample 4: Pituitary Tumor": ("data/test/pituitary", "Pituitary")
    }

    # Header
    st.markdown('<div class="main-header">🧠 NeuroScan-AI: Automated Brain Tumor MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Computer-Aided Diagnostic (CAD) system powered by Deep Convolutional Neural Networks and Transfer Learning</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔬 Diagnostic Inference", "📊 Model Performance & Benchmarks", "ℹ️ Academic Overview & System Info"])

    with tab1:
        col_input, col_result = st.columns([1, 1.2], gap="large")

        active_image = None
        image_source_desc = ""

        with col_input:
            st.markdown("### 1. Provide Brain MRI Scan")
            uploaded_file = st.file_uploader("Upload Axial Brain MRI Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

            if uploaded_file is not None:
                active_image = Image.open(uploaded_file).convert("RGB")
                image_source_desc = f"Uploaded File: {uploaded_file.name}"
            elif sample_choice != "None (Upload custom MRI)":
                folder_path, cat_name = sample_mapping[sample_choice]
                full_folder = os.path.join(os.path.dirname(__file__), folder_path)
                if os.path.exists(full_folder):
                    files = [f for f in os.listdir(full_folder) if f.endswith(('.jpg', '.png'))]
                    if files:
                        active_image = Image.open(os.path.join(full_folder, files[0])).convert("RGB")
                        image_source_desc = f"Benchmark Benchmark Scan ({cat_name})"

            if active_image is not None:
                st.image(active_image, caption=f"MRI Input: {image_source_desc}", use_container_width=True)
                st.info(f"**Image Dimensions**: {active_image.size[0]} x {active_image.size[1]} px | **Color Space**: {active_image.mode}")
            else:
                st.warning("Please upload a brain MRI scan or select a sample from the sidebar to run diagnostic analysis.")

        with col_result:
            st.markdown("### 2. Deep Learning Diagnostic Output")
            if predictor is None:
                st.error("Model weights file (`models/neuroscan_best_model.pt`) not detected. Please execute model training first.")
            elif active_image is not None:
                with st.spinner("Executing neural inference across 4 diagnostic classes..."):
                    result = predictor.predict(active_image)

                if result.get("success"):
                    pred_class = result["display_name"]
                    conf = result["confidence_score"]
                    color = result["badge_color"]
                    status = result["diagnostic_status"]
                    notes = result["clinical_notes"]
                    latency = result["inference_time_ms"]

                    st.markdown(f"""
                    <div style="background-color: {color};" class="status-badge">
                        {status.upper()}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"## **{pred_class}**")
                    st.metric("Model Confidence", f"{conf:.2f}%", delta=f"{latency:.1f} ms latency")
                    st.progress(conf / 100.0)

                    st.markdown("#### Probability Distribution Across Diagnostic Classes")
                    prob_df = pd.DataFrame([
                        {"Class": k, "Probability (%)": v}
                        for k, v in result["class_probabilities"].items()
                    ])

                    chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X('Probability (%):Q', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Class:N', sort='-x', title=""),
                        color=alt.Color('Class:N', legend=None, scale=alt.Scale(
                            domain=['Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor', 'Normal (No Tumor)'],
                            range=['#d62728', '#ff7f0e', '#9467bd', '#2ca02c']
                        )),
                        tooltip=['Class', 'Probability (%)']
                    ).properties(height=200)

                    st.altair_chart(chart, use_container_width=True)

                    st.markdown("#### Clinical Pathology Notes")
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>Clinical Assessment:</strong> {notes}<br/>
                        <small style="color: #718096;">*Automated assistance for neuro-radiologists. Must be correlated with clinical symptoms and contrast-enhanced imaging.</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(result.get("error", "Prediction failed."))
            else:
                st.info("Awaiting input MRI scan to generate automated classification and probability scores.")

    with tab2:
        st.markdown("### Quantitative Model Performance & Benchmark Results")
        
        metrics_path = os.path.join(os.path.dirname(__file__), "outputs", "test_metrics.json")
        training_path = os.path.join(os.path.dirname(__file__), "outputs", "training_metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                t_metrics = json.load(f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Test Accuracy", f"{t_metrics['overall_accuracy']}%")
            c2.metric("Macro F1-Score", f"{t_metrics['macro_f1']}")
            c3.metric("Weighted F1-Score", f"{t_metrics['weighted_f1']}")
            c4.metric("Avg Inference Latency", f"{t_metrics['avg_inference_latency_ms']} ms")

            st.markdown("#### Detailed Per-Class Evaluation Metrics")
            rows = []
            for cname, m in t_metrics["per_class_metrics"].items():
                rows.append({
                    "Clinical Category": m["display_name"],
                    "Precision": f"{m['precision']:.4f}",
                    "Recall": f"{m['recall']:.4f}",
                    "F1-Score": f"{m['f1_score']:.4f}",
                    "Test Samples (Support)": m["support"]
                })
            st.table(pd.DataFrame(rows))

        col_g1, col_g2 = st.columns(2)
        curve_img = os.path.join(os.path.dirname(__file__), "outputs", "accuracy_loss_curve.png")
        cm_img = os.path.join(os.path.dirname(__file__), "outputs", "confusion_matrix.png")

        with col_g1:
            if os.path.exists(curve_img):
                st.image(curve_img, caption="Figure 1: Training and Validation Loss & Accuracy Convergence Curves", use_container_width=True)
        with col_g2:
            if os.path.exists(cm_img):
                st.image(cm_img, caption="Figure 2: 4x4 Normalized Multiclass Confusion Matrix", use_container_width=True)

        samples_img = os.path.join(os.path.dirname(__file__), "outputs", "sample_predictions.png")
        if os.path.exists(samples_img):
            st.image(samples_img, caption="Figure 3: Sample Test Set Predictions with Ground Truth & Softmax Confidence", use_container_width=True)

    with tab3:
        st.markdown("### Academic Course Project Information")
        st.markdown("""
        - **Institution**: Prasad V Potluri Siddhartha Institute of Technology
        - **Department**: Department of Computer Science & Engineering
        - **Academic Year**: 2026-27
        - **Course**: Deep Learning Course Project
        - **Student / Developer**: Sumith Alanke (`sumithalanke46`)
        - **Project Title**: NeuroScan-AI: Multiclass Brain Tumor Classification from MRI Scans using Deep Transfer Learning
        
        #### Research Methodology & Architecture:
        1. **Data Ingestion**: Standardized 224x224 RGB axial brain MRI slices categorized into Glioma, Meningioma, Pituitary, and Normal.
        2. **Preprocessing**: Data augmentation via horizontal reflections, affine rotations ($\pm 15^\circ$), and ImageNet mean/standard-deviation normalization.
        3. **Deep Learning Backbone**: Pretrained MobileNetV2 architecture with depthwise separable convolutions and inverted residual blocks.
        4. **Regularization & Optimization**: Dropout regularization (0.3 / 0.2), Batch Normalization, Cross-Entropy Loss with AdamW optimization and Cosine Annealing learning rate schedule.
        """)

        st.warning("""
        **Clinical Disclaimer**: NeuroScan-AI is an experimental computer-aided diagnosis system developed for academic research and educational demonstration. It is not approved for autonomous primary medical diagnosis. Clinical decisions must always be corroborated by licensed medical practitioners and radiologists.
        """)


if __name__ == "__main__":
    main()
