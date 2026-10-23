"""
app.py — Streamlit Interactive Dashboard for ML Image Manipulation Checker.

A premium, dark-themed dashboard that lets users:
1. Upload images and see baseline CNN classification
2. Apply benign transformations or FGSM adversarial attacks
3. Observe how manipulations degrade model confidence
4. Apply spatial filtering defenses (3x3 Median, 3x3 Gaussian, JPEG, Bilateral) and see confidence recovery
5. NEW: Clean-Image Impact View — see how much the defense degrades a clean image
6. NEW: Side-by-Side Multi-Model Comparison — run the same attack/defense on ResNet-50 and MobileNetV2
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from models.classifier import load_model, preprocess_image, classify, denormalize_tensor
from utils.manipulation import BENIGN_TRANSFORMS, fgsm_attack
from utils.defense import DEFENSE_METHODS

# ============================================================================
# Page Configuration & Custom CSS
# ============================================================================

st.set_page_config(
    page_title="ML Image Manipulation Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- Import Modern Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global Styles ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Gradient Hero Title ---- */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7C3AED, #EC4899, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ---- Glassmorphism Cards ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
    }

    /* ---- Section Headers ---- */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---- Status Badges ---- */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-info {
        background: rgba(124, 58, 237, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }

    /* ---- Confidence Bar ---- */
    .conf-bar-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin-top: 4px;
        margin-bottom: 10px;
    }
    .conf-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ---- Metric Cards ---- */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---- Clean Impact Warning Banner ---- */
    .clean-impact-banner {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
    }
    .clean-impact-banner.safe {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.25);
    }

    /* ---- Model Comparison Card ---- */
    .model-card-header {
        font-size: 1rem;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    .resnet-header {
        background: rgba(124, 58, 237, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
    .mobilenet-header {
        background: rgba(236, 72, 153, 0.15);
        color: #F9A8D4;
        border: 1px solid rgba(236, 72, 153, 0.3);
    }

    /* ---- Sidebar Styling ---- */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }

    /* ---- Hide Streamlit Defaults ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    button[kind="header"] {visibility: visible !important;}

    /* ---- Smooth Animations ---- */
    .stImage, .stMetric, div[data-testid="column"] {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

def render_confidence_bars(predictions: list[dict], color: str = "#7C3AED"):
    """Render styled confidence bars for prediction results."""
    for pred in predictions:
        conf = pred["confidence"]
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.85rem; color:#E2E8F0; font-weight:500;">{pred['class_name']}</span>
            <span style="font-size:0.85rem; color:{color}; font-weight:700;">{conf:.1f}%</span>
        </div>
        <div class="conf-bar-container">
            <div class="conf-bar" style="width:{conf}%; background:linear-gradient(90deg, {color}, {color}88);"></div>
        </div>
        """, unsafe_allow_html=True)


def render_metric_cards(cards: list[dict]):
    """Render a row of styled metric cards."""
    html = '<div class="metric-row">'
    for card in cards:
        html += f"""
<div class="metric-card">
    <div class="metric-value" style="color:{card.get('color', '#7C3AED')}">{card['value']}</div>
    <div class="metric-label">{card['label']}</div>
</div>
"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def create_confidence_comparison_chart(
    baseline_preds: list[dict],
    manipulated_preds: list[dict],
    defended_preds: list[dict] | None = None,
) -> go.Figure:
    """Create a grouped bar chart comparing confidence across stages."""
    all_names = []
    seen = set()
    for preds in [baseline_preds, manipulated_preds] + ([defended_preds] if defended_preds else []):
        for p in preds:
            if p["class_name"] not in seen:
                all_names.append(p["class_name"])
                seen.add(p["class_name"])

    def get_conf(preds, name):
        for p in preds:
            if p["class_name"] == name:
                return p["confidence"]
        return 0.0

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Baseline", x=all_names,
                         y=[get_conf(baseline_preds, n) for n in all_names],
                         marker_color="#7C3AED", marker_line_width=0))
    fig.add_trace(go.Bar(name="After Manipulation", x=all_names,
                         y=[get_conf(manipulated_preds, n) for n in all_names],
                         marker_color="#EF4444", marker_line_width=0))
    if defended_preds:
        fig.add_trace(go.Bar(name="After Defense", x=all_names,
                             y=[get_conf(defended_preds, n) for n in all_names],
                             marker_color="#10B981", marker_line_width=0))

    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Confidence (%)", gridcolor="rgba(255,255,255,0.05)", range=[0, 100]),
        margin=dict(l=40, r=20, t=40, b=80),
        height=400,
    )
    return fig


def create_model_comparison_chart(
    resnet_data: dict,
    mobilenet_data: dict,
) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing ResNet-50 vs MobileNetV2
    across Baseline, Attacked, and Defended stages.
    """
    stages = ["Baseline", "After Attack", "After Defense"]
    resnet_vals = [
        resnet_data["baseline_conf"],
        resnet_data["attacked_conf"],
        resnet_data["defended_conf"] if resnet_data["defended_conf"] is not None else 0,
    ]
    mobilenet_vals = [
        mobilenet_data["baseline_conf"],
        mobilenet_data["attacked_conf"],
        mobilenet_data["defended_conf"] if mobilenet_data["defended_conf"] is not None else 0,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ResNet-50",
        x=stages,
        y=resnet_vals,
        marker_color="#7C3AED",
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in resnet_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="MobileNetV2",
        x=stages,
        y=mobilenet_vals,
        marker_color="#EC4899",
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in mobilenet_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Top-1 Confidence (%)", gridcolor="rgba(255,255,255,0.05)", range=[0, 110]),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    return fig


def run_pipeline_for_model(model, original_image, original_np, is_adversarial,
                            epsilon_val, transform_name_val, apply_def, defense_name_val):
    """
    Run classify → attack/benign → (optional defense) for a given model.
    Returns a dict with all predictions and numpy images.
    """
    input_tensor = preprocess_image(original_image)
    baseline_preds = classify(model, input_tensor, top_k=5)
    top_pred = baseline_preds[0]

    if is_adversarial:
        adv_tensor = fgsm_attack(model, input_tensor, top_pred["class_index"], epsilon=epsilon_val)
        manipulated_np = denormalize_tensor(adv_tensor)
        manipulated_preds = classify(model, adv_tensor, top_k=5)
    else:
        transform_fn = BENIGN_TRANSFORMS[transform_name_val]
        manipulated_np = transform_fn(original_np)
        manipulated_tensor = preprocess_image(Image.fromarray(manipulated_np))
        manipulated_preds = classify(model, manipulated_tensor, top_k=5)

    defended_preds = None
    defended_np = None
    clean_defended_preds = None

    if apply_def:
        defense_fn = DEFENSE_METHODS[defense_name_val]
        defended_np_arr = defense_fn(manipulated_np)
        defended_tensor = preprocess_image(Image.fromarray(defended_np_arr))
        defended_preds = classify(model, defended_tensor, top_k=5)
        defended_np = defended_np_arr

        # Clean-Image Impact: apply the same defense filter to the ORIGINAL clean image
        clean_filtered_np = defense_fn(original_np)
        clean_filtered_tensor = preprocess_image(Image.fromarray(clean_filtered_np))
        clean_defended_preds = classify(model, clean_filtered_tensor, top_k=5)

    return {
        "baseline_preds": baseline_preds,
        "manipulated_preds": manipulated_preds,
        "defended_preds": defended_preds,
        "clean_defended_preds": clean_defended_preds,
        "manipulated_np": manipulated_np,
        "defended_np": defended_np,
        "top_pred": top_pred,
    }


# ============================================================================
# Model Caching
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_model(model_name: str):
    """Load and cache the ML model."""
    return load_model(model_name)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    # Mode toggle
    compare_mode = st.toggle("🔁 Multi-Model Comparison Mode", value=False,
                              help="Run the same attack and defense on both ResNet-50 and MobileNetV2 simultaneously.")

    if not compare_mode:
        model_choice = st.selectbox(
            "🧠 Select Model",
            ["resnet50", "mobilenetv2"],
            format_func=lambda x: "ResNet-50" if x == "resnet50" else "MobileNet V2",
        )

    st.divider()

    st.markdown('<div class="section-header">🎯 Manipulation</div>', unsafe_allow_html=True)

    manipulation_type = st.radio(
        "Attack Type",
        ["Benign Transform", "FGSM Adversarial Attack"],
        label_visibility="collapsed",
    )

    if manipulation_type == "Benign Transform":
        transform_name = st.selectbox(
            "Select Transform",
            list(BENIGN_TRANSFORMS.keys()),
        )
        epsilon = 0.03  # unused but defined to avoid NameError
    else:
        transform_name = list(BENIGN_TRANSFORMS.keys())[0]  # unused
        epsilon = st.slider(
            "FGSM Epsilon (ε)",
            min_value=0.01,
            max_value=0.08,
            value=0.03,
            step=0.01,
            help="Perturbation magnitude. Range in paper: ε = 0.01 to 0.08.",
        )

    st.divider()

    st.markdown('<div class="section-header">🛡️ Defense</div>', unsafe_allow_html=True)

    apply_defense = st.checkbox("Apply Defense", value=True)

    if apply_defense:
        defense_name = st.selectbox(
            "Select Defense Method",
            list(DEFENSE_METHODS.keys()),
        )
    else:
        defense_name = list(DEFENSE_METHODS.keys())[0]  # unused

    if apply_defense:
        show_clean_impact = st.checkbox(
            "🔍 Show Clean-Image Impact",
            value=True,
            help="Applies the same defense filter to the original clean image to show how much accuracy it degrades without any attack.",
        )
    else:
        show_clean_impact = False

    st.divider()

    st.markdown("""
    <div style="
        padding: 1rem;
        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.6;
    ">
        <strong style="color: #A78BFA;">How it works:</strong><br>
        Upload an image → see baseline classification →
        apply benign or adversarial manipulations →
        observe confidence shifts → apply defense filters
        to recover predictions.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Content
# ============================================================================

st.markdown('<div class="hero-title">ML Image Manipulation Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Explore how image manipulations and adversarial attacks affect deep learning classifiers — and how defenses can recover.'
    '</div>',
    unsafe_allow_html=True,
)

# Upload area
uploaded_file = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(original_image)
    is_adversarial = manipulation_type == "FGSM Adversarial Attack"

    # ============================================================================
    # MODE 1: SINGLE MODEL
    # ============================================================================
    if not compare_mode:
        with st.spinner(f"Loading {model_choice}..."):
            model = get_model(model_choice)

        result = run_pipeline_for_model(
            model, original_image, original_np,
            is_adversarial, epsilon, transform_name,
            apply_defense, defense_name,
        )

        baseline_preds    = result["baseline_preds"]
        manipulated_preds = result["manipulated_preds"]
        defended_preds    = result["defended_preds"]
        clean_def_preds   = result["clean_defended_preds"]
        manipulated_np    = result["manipulated_np"]
        defended_np       = result["defended_np"]
        top_pred          = result["top_pred"]

        # ---- Baseline ----
        st.markdown('<div class="section-header">📸 Baseline Classification</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="margin-bottom:1rem;">
                <span class="badge badge-success">Top Prediction</span>
                <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#E2E8F0;">{top_pred['class_name']}</span>
                <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#10B981;">{top_pred['confidence']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            render_confidence_bars(baseline_preds, color="#7C3AED")

        st.divider()

        # ---- Manipulation ----
        if is_adversarial:
            st.markdown('<div class="section-header">⚡ FGSM Adversarial Attack</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section-header">🔄 Benign Transform: {transform_name}</div>', unsafe_allow_html=True)

        manip_top = manipulated_preds[0]
        attack_success = manip_top["class_name"] != top_pred["class_name"]
        confidence_drop = top_pred["confidence"] - manip_top["confidence"]

        if is_adversarial:
            badge_class = "badge-danger" if attack_success else "badge-warning"
            badge_text = "ATTACK SUCCEEDED" if attack_success else "ATTACK RESISTED"
        else:
            badge_class = "badge-warning" if abs(confidence_drop) > 5 else "badge-info"
            badge_text = "CLASSIFICATION CHANGED" if attack_success else "CLASSIFICATION STABLE"

        col3, col4 = st.columns([1, 1.2])
        with col3:
            st.image(manipulated_np, caption="Manipulated Image", use_container_width=True)
        with col4:
            st.markdown(f"""
            <div style="margin-bottom:1rem;">
                <span class="badge {badge_class}">{badge_text}</span>
                <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#E2E8F0;">{manip_top['class_name']}</span>
                <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#EF4444;">{manip_top['confidence']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            render_confidence_bars(manipulated_preds, color="#EF4444")

        render_metric_cards([
            {"value": f"{top_pred['confidence']:.1f}%", "label": "Baseline Confidence", "color": "#7C3AED"},
            {"value": f"{manip_top['confidence']:.1f}%", "label": "After Manipulation", "color": "#EF4444"},
            {"value": f"{confidence_drop:+.1f}%", "label": "Confidence Shift",
             "color": "#FBBF24" if abs(confidence_drop) > 5 else "#10B981"},
        ])

        # ---- Defense ----
        if apply_defense:
            st.divider()
            st.markdown(f'<div class="section-header">🛡️ Defense Applied: {defense_name}</div>', unsafe_allow_html=True)

            def_top = defended_preds[0]
            recovered = def_top["class_name"] == top_pred["class_name"]
            recovery_badge = "badge-success" if recovered else "badge-warning"
            recovery_text = "CLASSIFICATION RECOVERED" if recovered else "PARTIALLY RECOVERED"
            recovery_delta = def_top["confidence"] - manip_top["confidence"]

            col5, col6 = st.columns([1, 1.2])
            with col5:
                st.image(defended_np, caption="Defended Image", use_container_width=True)
            with col6:
                st.markdown(f"""
                <div style="margin-bottom:1rem;">
                    <span class="badge {recovery_badge}">{recovery_text}</span>
                    <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#E2E8F0;">{def_top['class_name']}</span>
                    <span style="margin-left:0.5rem; font-size:1.1rem; font-weight:700; color:#10B981;">{def_top['confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                render_confidence_bars(defended_preds, color="#10B981")

            render_metric_cards([
                {"value": f"{manip_top['confidence']:.1f}%", "label": "Before Defense", "color": "#EF4444"},
                {"value": f"{def_top['confidence']:.1f}%", "label": "After Defense", "color": "#10B981"},
                {"value": f"{recovery_delta:+.1f}%", "label": "Recovery Delta",
                 "color": "#10B981" if recovery_delta > 0 else "#EF4444"},
            ])

            # ---- Clean-Image Impact View ----
            if show_clean_impact and clean_def_preds is not None:
                st.divider()
                st.markdown('<div class="section-header">🔍 Clean-Image Impact View</div>', unsafe_allow_html=True)

                clean_top_after_filter = clean_def_preds[0]
                clean_conf_before = top_pred["confidence"]
                clean_conf_after  = clean_top_after_filter["confidence"]
                clean_drop        = clean_conf_before - clean_conf_after
                class_changed     = clean_top_after_filter["class_name"] != top_pred["class_name"]

                # Banner
                if abs(clean_drop) <= 3 and not class_changed:
                    banner_class = "clean-impact-banner safe"
                    impact_icon  = "✅"
                    impact_msg   = f"The <strong>{defense_name}</strong> filter is <strong>safe for clean images</strong>. It causes only a negligible confidence shift of <strong>{clean_drop:+.1f}%</strong> without an attack."
                elif class_changed:
                    banner_class = "clean-impact-banner"
                    impact_icon  = "⚠️"
                    impact_msg   = f"The <strong>{defense_name}</strong> filter <strong>changes the classification</strong> of a clean image from <em>{top_pred['class_name']}</em> to <em>{clean_top_after_filter['class_name']}</em>. This defense introduces its own accuracy cost."
                else:
                    banner_class = "clean-impact-banner"
                    impact_icon  = "⚠️"
                    impact_msg   = f"The <strong>{defense_name}</strong> filter degrades confidence by <strong>{clean_drop:+.1f}%</strong> on a clean image. Consider a lighter filter to reduce this side-effect."

                st.markdown(f"""
                <div class="{banner_class}">
                    <span style="font-size:1.2rem;">{impact_icon}</span>
                    <span style="font-size:0.9rem; color:#E2E8F0; margin-left:0.5rem;">{impact_msg}</span>
                </div>
                """, unsafe_allow_html=True)

                # Metric row
                render_metric_cards([
                    {"value": f"{clean_conf_before:.1f}%", "label": "Clean Image Confidence", "color": "#7C3AED"},
                    {"value": f"{clean_conf_after:.1f}%",  "label": f"After {defense_name}", "color": "#F59E0B"},
                    {"value": f"{clean_drop:+.1f}%",       "label": "Filter Degradation",
                     "color": "#EF4444" if abs(clean_drop) > 5 else "#10B981"},
                ])

                # Compact bar comparison
                st.markdown("**Top-5 predictions on clean image after filtering:**")
                render_confidence_bars(clean_def_preds, color="#F59E0B")

        # ---- Comparison Chart ----
        st.divider()
        st.markdown('<div class="section-header">📊 Confidence Comparison</div>', unsafe_allow_html=True)
        fig = create_confidence_comparison_chart(baseline_preds, manipulated_preds, defended_preds)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ============================================================================
    # MODE 2: SIDE-BY-SIDE MULTI-MODEL COMPARISON
    # ============================================================================
    else:
        st.markdown('<div class="section-header">🔁 Multi-Model Comparison: ResNet-50 vs MobileNetV2</div>', unsafe_allow_html=True)
        st.caption(f"Running the same pipeline on both models simultaneously — Attack: **{manipulation_type}** | Defense: **{defense_name if apply_defense else 'None'}**")

        with st.spinner("Loading both models and running pipeline..."):
            model_rn  = get_model("resnet50")
            model_mn  = get_model("mobilenetv2")
            result_rn = run_pipeline_for_model(model_rn, original_image, original_np,
                                               is_adversarial, epsilon, transform_name,
                                               apply_defense, defense_name)
            result_mn = run_pipeline_for_model(model_mn, original_image, original_np,
                                               is_adversarial, epsilon, transform_name,
                                               apply_defense, defense_name)

        # ---- Header Row ----
        col_rn_h, col_mn_h = st.columns(2)
        with col_rn_h:
            st.markdown('<div class="model-card-header resnet-header">🟣 ResNet-50</div>', unsafe_allow_html=True)
        with col_mn_h:
            st.markdown('<div class="model-card-header mobilenet-header">🩷 MobileNetV2</div>', unsafe_allow_html=True)

        st.divider()

        # ---- Baseline Row ----
        st.markdown("**📸 Baseline Classification**")
        col1, col2 = st.columns(2)
        with col1:
            tp_rn = result_rn["top_pred"]
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <span class="badge badge-success">Top-1</span>
                <span style="margin-left:0.4rem; font-weight:700; color:#E2E8F0;">{tp_rn['class_name']}</span>
                <span style="margin-left:0.4rem; color:#10B981; font-weight:700;">{tp_rn['confidence']:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            render_confidence_bars(result_rn["baseline_preds"][:3], color="#7C3AED")
        with col2:
            tp_mn = result_mn["top_pred"]
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <span class="badge badge-success">Top-1</span>
                <span style="margin-left:0.4rem; font-weight:700; color:#E2E8F0;">{tp_mn['class_name']}</span>
                <span style="margin-left:0.4rem; color:#10B981; font-weight:700;">{tp_mn['confidence']:.1f}%</span>
            </div>""", unsafe_allow_html=True)
            render_confidence_bars(result_mn["baseline_preds"][:3], color="#EC4899")

        st.divider()

        # ---- Attacked Row ----
        attack_label = f"⚡ FGSM (ε={epsilon})" if is_adversarial else f"🔄 {transform_name}"
        st.markdown(f"**{attack_label}**")
        col3, col4 = st.columns(2)
        with col3:
            mt_rn = result_rn["manipulated_preds"][0]
            drop_rn = tp_rn["confidence"] - mt_rn["confidence"]
            ok_rn = mt_rn["class_name"] != tp_rn["class_name"]
            st.image(result_rn["manipulated_np"], caption="ResNet-50 — Manipulated", use_container_width=True)
            render_metric_cards([
                {"value": f"{mt_rn['confidence']:.1f}%", "label": "Attacked Conf.", "color": "#EF4444"},
                {"value": f"{drop_rn:+.1f}%", "label": "Drop", "color": "#EF4444" if abs(drop_rn) > 5 else "#10B981"},
                {"value": "❌ FOOLED" if ok_rn else "✅ HELD", "label": "Attack Result",
                 "color": "#EF4444" if ok_rn else "#10B981"},
            ])
        with col4:
            mt_mn = result_mn["manipulated_preds"][0]
            drop_mn = tp_mn["confidence"] - mt_mn["confidence"]
            ok_mn = mt_mn["class_name"] != tp_mn["class_name"]
            st.image(result_mn["manipulated_np"], caption="MobileNetV2 — Manipulated", use_container_width=True)
            render_metric_cards([
                {"value": f"{mt_mn['confidence']:.1f}%", "label": "Attacked Conf.", "color": "#EF4444"},
                {"value": f"{drop_mn:+.1f}%", "label": "Drop", "color": "#EF4444" if abs(drop_mn) > 5 else "#10B981"},
                {"value": "❌ FOOLED" if ok_mn else "✅ HELD", "label": "Attack Result",
                 "color": "#EF4444" if ok_mn else "#10B981"},
            ])

        # ---- Defense Row ----
        if apply_defense:
            st.divider()
            st.markdown(f"**🛡️ After Defense: {defense_name}**")
            col5, col6 = st.columns(2)
            with col5:
                df_rn = result_rn["defended_preds"][0]
                rd_rn = df_rn["confidence"] - mt_rn["confidence"]
                rec_rn = df_rn["class_name"] == tp_rn["class_name"]
                st.image(result_rn["defended_np"], caption="ResNet-50 — Defended", use_container_width=True)
                render_metric_cards([
                    {"value": f"{df_rn['confidence']:.1f}%", "label": "Defended Conf.", "color": "#10B981"},
                    {"value": f"{rd_rn:+.1f}%", "label": "Recovery Delta", "color": "#10B981" if rd_rn > 0 else "#EF4444"},
                    {"value": "✅ RECOVERED" if rec_rn else "⚠️ PARTIAL", "label": "Recovery",
                     "color": "#10B981" if rec_rn else "#FBBF24"},
                ])
            with col6:
                df_mn = result_mn["defended_preds"][0]
                rd_mn = df_mn["confidence"] - mt_mn["confidence"]
                rec_mn = df_mn["class_name"] == tp_mn["class_name"]
                st.image(result_mn["defended_np"], caption="MobileNetV2 — Defended", use_container_width=True)
                render_metric_cards([
                    {"value": f"{df_mn['confidence']:.1f}%", "label": "Defended Conf.", "color": "#10B981"},
                    {"value": f"{rd_mn:+.1f}%", "label": "Recovery Delta", "color": "#10B981" if rd_mn > 0 else "#EF4444"},
                    {"value": "✅ RECOVERED" if rec_mn else "⚠️ PARTIAL", "label": "Recovery",
                     "color": "#10B981" if rec_mn else "#FBBF24"},
                ])

            # ---- Clean-Image Impact for both models ----
            if show_clean_impact:
                st.divider()
                st.markdown('<div class="section-header">🔍 Clean-Image Filter Impact (Both Models)</div>', unsafe_allow_html=True)
                col7, col8 = st.columns(2)
                with col7:
                    cdf_rn = result_rn["clean_defended_preds"][0] if result_rn["clean_defended_preds"] else None
                    if cdf_rn:
                        clean_drop_rn = tp_rn["confidence"] - cdf_rn["confidence"]
                        render_metric_cards([
                            {"value": f"{tp_rn['confidence']:.1f}%", "label": "Clean Baseline", "color": "#7C3AED"},
                            {"value": f"{cdf_rn['confidence']:.1f}%", "label": "After Filter", "color": "#F59E0B"},
                            {"value": f"{clean_drop_rn:+.1f}%", "label": "Filter Cost (ResNet)",
                             "color": "#EF4444" if abs(clean_drop_rn) > 5 else "#10B981"},
                        ])
                with col8:
                    cdf_mn = result_mn["clean_defended_preds"][0] if result_mn["clean_defended_preds"] else None
                    if cdf_mn:
                        clean_drop_mn = tp_mn["confidence"] - cdf_mn["confidence"]
                        render_metric_cards([
                            {"value": f"{tp_mn['confidence']:.1f}%", "label": "Clean Baseline", "color": "#EC4899"},
                            {"value": f"{cdf_mn['confidence']:.1f}%", "label": "After Filter", "color": "#F59E0B"},
                            {"value": f"{clean_drop_mn:+.1f}%", "label": "Filter Cost (MobileNet)",
                             "color": "#EF4444" if abs(clean_drop_mn) > 5 else "#10B981"},
                        ])

        # ---- Aggregated Comparison Chart ----
        st.divider()
        st.markdown('<div class="section-header">📊 ResNet-50 vs MobileNetV2 — Pipeline Summary</div>', unsafe_allow_html=True)
        rn_chart_data = {
            "baseline_conf": tp_rn["confidence"],
            "attacked_conf": mt_rn["confidence"],
            "defended_conf": result_rn["defended_preds"][0]["confidence"] if apply_defense and result_rn["defended_preds"] else None,
        }
        mn_chart_data = {
            "baseline_conf": tp_mn["confidence"],
            "attacked_conf": mt_mn["confidence"],
            "defended_conf": result_mn["defended_preds"][0]["confidence"] if apply_defense and result_mn["defended_preds"] else None,
        }
        fig_compare = create_model_comparison_chart(rn_chart_data, mn_chart_data)
        st.plotly_chart(fig_compare, use_container_width=True, config={"displayModeBar": False})

        # Insight box
        diff_attacked = abs(mt_rn["confidence"] - mt_mn["confidence"])
        st.markdown(f"""
        <div style="
            padding:1rem 1.4rem;
            background:rgba(124,58,237,0.06);
            border:1px solid rgba(124,58,237,0.2);
            border-radius:12px;
            font-size:0.88rem;
            color:#CBD5E1;
            line-height:1.7;
        ">
            <strong style="color:#A78BFA;">💡 Insight:</strong>
            Under the same <em>{manipulation_type}</em> conditions, ResNet-50 retained
            <strong style="color:#7C3AED;">{mt_rn['confidence']:.1f}%</strong> confidence while
            MobileNetV2 retained <strong style="color:#EC4899;">{mt_mn['confidence']:.1f}%</strong>
            — a difference of <strong>{diff_attacked:.1f}%</strong>.
            This reflects how deeper residual architectures distribute gradient perturbations across more feature
            pathways than lightweight inverted-residual bottlenecks.
        </div>
        """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="
        text-align: center;
        padding: 5rem 2rem;
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        margin-top: 2rem;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🖼️</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.5rem;">
            Upload an Image to Begin
        </div>
        <div style="font-size: 0.95rem; color: #64748B; max-width: 500px; margin: 0 auto; line-height: 1.7;">
            Drag and drop or click to upload a JPG, PNG, or WebP image.
            The system will classify it using a pre-trained CNN, then demonstrate
            how manipulations and adversarial attacks affect the predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)
