import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Industrial AI - Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0a0a;
        --bg-secondary: #0f0f0f;
        --bg-tertiary: #1a1a1a;
        --bg-card: #141414;
        --bg-elevated: #1f1f1f;
        --border-color: #2a2a2a;
        --border-subtle: #1f1f1f;
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --text-muted: #666666;
        --accent-cyan: #00d4ff;
        --accent-blue: #0099ff;
        --accent-green: #00ff88;
        --accent-orange: #ff9500;
        --accent-red: #ff3b30;
        --gradient-cyan: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        --gradient-success: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        --gradient-warning: linear-gradient(135deg, #ff9500 0%, #ff6200 100%);
        --gradient-danger: linear-gradient(135deg, #ff3b30 0%, #d70015 100%);
        --gradient-card: linear-gradient(145deg, #141414 0%, #1a1a1a 100%);
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 16px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
        --shadow-xl: 0 16px 48px rgba(0,0,0,0.6);
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Header */
    .main-header {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-cyan);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: var(--gradient-cyan);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        text-align: center;
        font-weight: 400;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    
    .css-1d391kg .css-17eq0hr {
        background: var(--bg-secondary);
    }
    
    /* Cards */
    .card {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--accent-cyan);
    }
    
    /* Status Indicators */
    .status-normal {
        background: var(--gradient-success);
        color: var(--bg-primary);
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .status-warning {
        background: var(--gradient-warning);
        color: var(--bg-primary);
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(255, 149, 0, 0.3);
    }
    
    .status-critical {
        background: var(--gradient-danger);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
        position: relative;
        overflow: hidden;
        animation: critical-pulse 2s infinite;
        box-shadow: 0 0 30px rgba(255, 59, 48, 0.4);
    }
    
    @keyframes critical-pulse {
        0%, 100% { 
            box-shadow: 0 0 30px rgba(255, 59, 48, 0.4);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 0 50px rgba(255, 59, 48, 0.6);
            transform: scale(1.02);
        }
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-cyan);
        color: var(--bg-primary);
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: var(--gradient-cyan);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover::after {
        transform: scaleX(1);
    }
    
    /* Input Elements */
    .stSelectbox > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    .stSlider > div > div > div {
        background: var(--accent-cyan);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: var(--shadow-lg);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-cyan);
    }
    
    /* DataFrames */
    .dataframe {
        background: var(--bg-card);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    .dataframe th {
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .dataframe td {
        color: var(--text-secondary);
        border-top: 1px solid var(--border-subtle);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }
    
    /* Glow Effects */
    .glow-cyan {
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .glow-green {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    }
    
    .glow-orange {
        box-shadow: 0 0 20px rgba(255, 149, 0, 0.3);
    }
    
    /* Text Colors */
    .stMarkdown p {
        color: var(--text-secondary);
    }
    
    .stSelectbox label, .stSlider label {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        background: var(--gradient-card);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .footer h3 {
        background: var(--gradient-cyan);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 1rem 0;
    }
    
    .footer p {
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }
    
    /* Animations */
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .slide-up {
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import os
    import joblib
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try pipeline models first
        model_path = os.path.join(current_dir, 'Model_file', 'best_model_final.pkl')
        scaler_path = os.path.join(current_dir, 'Model_file', 'scaler_final.pkl')
        metadata_path = os.path.join(current_dir, 'Notebooks', 'pipeline_metadata.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            metadata = {}
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                f1_score = metadata.get('test_metrics', {}).get('f1_weighted', 0)
                st.success(f"✅ {metadata.get('best_model', 'Model')} | F1: {f1_score:.4f}")
                
                with st.expander("🏆 Pipeline Results"):
                    st.write(f"**Best Algorithm:** {metadata.get('best_model', 'Unknown')}")
                    
                    metrics = metadata.get('test_metrics', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("F1 Weighted", f"{metrics.get('f1_weighted', 0):.4f}")
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                        st.metric("Precision", f"{metrics.get('precision_weighted', 0):.4f}")
                    with col2:
                        st.metric("F1 Macro", f"{metrics.get('f1_macro', 0):.4f}")
                        st.metric("Recall", f"{metrics.get('recall_weighted', 0):.4f}")
                        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
                    
                    if 'all_results' in metadata:
                        st.write("**All Model Rankings:**")
                        all_results = metadata['all_results']
                        sorted_models = sorted(all_results.items(), 
                                             key=lambda x: x[1].get('f1_weighted', 0), 
                                             reverse=True)
                        for i, (name, metrics) in enumerate(sorted_models[:5], 1):
                            st.write(f"{i}. {name}: {metrics.get('f1_weighted', 0):.4f}")
            
            return model, scaler, metadata
        
        st.warning("⚠️ Run ml_pipeline.py to train models")
        return None, None, {}
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, {}

def main():
    # Modern Header
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🏭 Industrial AI</h1>
        <p>Predictive Maintenance System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler, metadata = load_model()
    
    if model is None:
        st.stop()
    
    # Get failure mapping from metadata or use new default
    failure_mapping = metadata.get('failure_mapping', {
        -1: 'No Failure',
        0: 'Heat Dissipation Failure',
        1: 'Overstrain Failure', 
        2: 'Power Failure',
        3: 'Random Failure',
        4: 'Tool Wear Failure'
    })
    
    # Modern Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: var(--gradient-card); border-radius: 16px; margin-bottom: 1.5rem; border: 1px solid var(--border-color);">
        <h2 style="color: var(--text-primary); margin: 0; font-size: 1.3rem;">⚙️ Control Panel</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">System Parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Scenarios
    st.sidebar.markdown("""
    <div style="background: var(--bg-card); border-left: 3px solid var(--accent-cyan); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: var(--accent-cyan); margin: 0 0 0.5rem 0; font-size: 1rem;">🎯 Quick Scenarios</h3>
        <p style="color: var(--text-secondary); font-size: 0.85rem; margin: 0;">Test configurations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.caption("⚠️ Model detects Tool Wear & Power failures")
    
    if st.sidebar.button("🔧 Tool Wear Failure", use_container_width=True):
        st.session_state.machine_type = 'Low'
        st.session_state.air_temp = 300.0
        st.session_state.process_temp = 310.0
        st.session_state.rotational_speed = 1500
        st.session_state.torque = 40.0
        st.session_state.tool_wear = 280
    
    if st.sidebar.button("⚡ Power Failure", use_container_width=True):
        st.session_state.machine_type = 'Low'
        st.session_state.air_temp = 298.0
        st.session_state.process_temp = 308.0
        st.session_state.rotational_speed = 2500
        st.session_state.torque = 60.0
        st.session_state.tool_wear = 100
    
    if st.sidebar.button("✅ Optimal", use_container_width=True):
        st.session_state.machine_type = 'High'
        st.session_state.air_temp = 305.0
        st.session_state.process_temp = 315.0
        st.session_state.rotational_speed = 1200
        st.session_state.torque = 15.0
        st.session_state.tool_wear = 50
    
    if st.sidebar.button("🔄 Reset", use_container_width=True):
        st.session_state.machine_type = 'Low'
        st.session_state.air_temp = 298.0
        st.session_state.process_temp = 308.0
        st.session_state.rotational_speed = 2000
        st.session_state.torque = 30.0
        st.session_state.tool_wear = 50

    type_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    machine_type = st.sidebar.selectbox("Machine Type", ['Low', 'Medium', 'High'], 
                                       index=['Low', 'Medium', 'High'].index(st.session_state.get('machine_type', 'Low')))
    
    air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 
                                st.session_state.get('air_temp', 298.0), 0.1)
    process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 
                                    st.session_state.get('process_temp', 308.0), 0.1)
    rotational_speed = st.sidebar.slider("Rotational Speed (rpm)", 1000, 3000, 
                                        st.session_state.get('rotational_speed', 2000), 10)
    torque = st.sidebar.slider("Torque (Nm)", 10.0, 80.0, 
                              st.session_state.get('torque', 30.0), 0.5)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 300, 
                                 st.session_state.get('tool_wear', 50), 1)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card slide-up">
            <h2 style="margin: 0 0 1rem 0;">📊 System Monitoring</h2>
            <p style="margin: 0; color: var(--text-secondary);">Real-time parameters</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear', 'Status'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Gauge colors
        gauge_colors = {
            'air_temp': '#00d4ff',
            'process_temp': '#0099ff', 
            'rpm': '#00ff88',
            'torque': '#ff9500',
            'tool_wear': '#00d4ff',
            'status': '#00ff88'
        }
        
        # Add gauges
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=air_temp,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [295, 305], 'tickcolor': '#666666'},
                'bar': {'color': gauge_colors['air_temp']},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [295, 300], 'color': 'rgba(0, 212, 255, 0.1)'},
                    {'range': [300, 305], 'color': 'rgba(255, 149, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#00d4ff", 'width': 2},
                    'thickness': 0.5,
                    'value': 302
                }
            }), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=process_temp,
            gauge={
                'axis': {'range': [305, 315], 'tickcolor': '#666666'},
                'bar': {'color': gauge_colors['process_temp']},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [305, 310], 'color': 'rgba(0, 255, 136, 0.1)'},
                    {'range': [310, 315], 'color': 'rgba(255, 59, 48, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#0099ff", 'width': 2},
                    'thickness': 0.5,
                    'value': 312
                }
            }), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=rotational_speed,
            gauge={
                'axis': {'range': [1000, 3000], 'tickcolor': '#666666'},
                'bar': {'color': gauge_colors['rpm']},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [1000, 2000], 'color': 'rgba(0, 255, 136, 0.1)'},
                    {'range': [2000, 3000], 'color': 'rgba(255, 149, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#00ff88", 'width': 2},
                    'thickness': 0.5,
                    'value': 2500
                }
            }), row=1, col=3)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=torque,
            gauge={
                'axis': {'range': [10, 80], 'tickcolor': '#666666'},
                'bar': {'color': gauge_colors['torque']},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [10, 50], 'color': 'rgba(0, 255, 136, 0.1)'},
                    {'range': [50, 80], 'color': 'rgba(255, 59, 48, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#ff9500", 'width': 2},
                    'thickness': 0.5,
                    'value': 60
                }
            }), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=tool_wear,
            gauge={
                'axis': {'range': [0, 300], 'tickcolor': '#666666'},
                'bar': {'color': gauge_colors['tool_wear']},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [0, 150], 'color': 'rgba(0, 255, 136, 0.1)'},
                    {'range': [150, 250], 'color': 'rgba(255, 149, 0, 0.1)'},
                    {'range': [250, 300], 'color': 'rgba(255, 59, 48, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#00d4ff", 'width': 2},
                    'thickness': 0.5,
                    'value': 250
                }
            }), row=2, col=2)
        
        # Status indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=100,
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#666666'},
                'bar': {'color': '#00ff88'},
                'bgcolor': '#141414',
                'borderwidth': 1,
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 59, 48, 0.1)'},
                    {'range': [50, 100], 'color': 'rgba(0, 255, 136, 0.1)'}
                ]
            }), row=2, col=3)
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            font={'color': '#ffffff', 'family': 'Inter'},
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameters summary
        st.markdown("### 📋 Current Parameters")
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.metric("Machine Type", machine_type)
            st.metric("Air Temperature", f"{air_temp} K")
            st.metric("Process Temperature", f"{process_temp} K")
        
        with param_col2:
            st.metric("Rotational Speed", f"{rotational_speed} rpm")
            st.metric("Torque", f"{torque} Nm")
            st.metric("Tool Wear", f"{tool_wear} min")
        
        # System status
        st.markdown("### 🔍 System Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            temp_status = "🟢 Normal" if air_temp < 302 else "🟡 High" if air_temp < 304 else "🔴 Critical"
            st.markdown(f"**Temperature:** {temp_status}")
            
            rpm_status = "🟢 Normal" if rotational_speed < 2500 else "🟡 High"
            st.markdown(f"**RPM:** {rpm_status}")
        
        with status_col2:
            torque_status = "🟢 Normal" if torque < 60 else "🟡 High" if torque < 75 else "🔴 Critical"
            st.markdown(f"**Torque:** {torque_status}")
            
            wear_status = "🟢 Good" if tool_wear < 150 else "🟡 Monitor" if tool_wear < 250 else "🔴 Replace"
            st.markdown(f"**Tool Wear:** {wear_status}")
    
    with col2:
        st.markdown("""
        <div class="card slide-up">
            <h2 style="margin: 0 0 1rem 0;">🔮 Failure Prediction</h2>
            <p style="margin: 0; color: var(--text-secondary);">AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze System", type="primary", use_container_width=True):
            # Prepare input data
            numerical_features = np.array([[
                air_temp,
                process_temp,
                rotational_speed,
                torque,
                tool_wear
            ]])
            
            # Scale features
            numerical_scaled = scaler.transform(numerical_features)
            
            # Combine features
            input_data = np.array([[
                type_mapping[machine_type],
                numerical_scaled[0][0],
                numerical_scaled[0][1],
                numerical_scaled[0][2],
                numerical_scaled[0][3],
                numerical_scaled[0][4]
            ]])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Failure type mapping
            failure_mapping = {
                0: 'No Failure',
                1: 'HEAT DISSIPATION FAILURE',
                2: 'OVERSTRAIN FAILURE', 
                3: 'POWER FAILURE',
                4: 'RANDOM FAILURE',
                5: 'TOOL WEAR FAILURE'
            }
            
            # Get prediction
            max_prob_index = np.argmax(probability)
            predicted_failure = failure_mapping.get(max_prob_index, 'Unknown Failure')
            max_prob = probability[max_prob_index]
            
            # Display results
            if predicted_failure == 'No Failure':
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="status-normal">
                        ✅ SYSTEM OPERATIONAL
                    </div>
                    <h3 style="color: var(--accent-green); margin: 1rem 0;">No Maintenance Required</h3>
                    <p style="color: var(--text-secondary);">All parameters within normal range</p>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 1rem;">Confidence: {max_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "Low"
            else:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="status-critical">
                        🚨 {predicted_failure}
                    </div>
                    <h3 style="color: var(--accent-red); margin: 1rem 0;">Immediate Action Required</h3>
                    <p style="color: var(--text-secondary);">System requires attention</p>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 1rem;">Confidence: {max_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "Critical"
            
            st.markdown("---")
            
            # Prediction details
            st.subheader("🔍 Analysis Details")
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.metric("Failure Type", predicted_failure)
                st.metric("Class ID", str(max_prob_index))
            
            with col_pred2:
                st.metric("Confidence", f"{max_prob:.1%}")
                st.metric("Risk Level", risk_level)
            
            # Probability distribution
            st.subheader("📈 Probability Distribution")
            
            prob_data = []
            for i, (class_id, class_name) in enumerate(failure_mapping.items()):
                if i < len(probability):
                    prob_data.append({
                        'Failure Type': class_name,
                        'Probability': probability[i],
                        'Class ID': class_id
                    })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            # Bar chart
            fig_prob = px.bar(prob_df, x='Failure Type', y='Probability', 
                             color='Probability', color_continuous_scale='Blues',
                             title="Failure Probability Distribution")
            fig_prob.update_xaxes(tickangle=45)
            fig_prob.update_layout(
                height=350, 
                showlegend=False,
                paper_bgcolor='#0a0a0a',
                plot_bgcolor='#0a0a0a',
                font={'color': '#ffffff'}
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Top predictions
            st.subheader("🏆 Top Predictions")
            top_3 = prob_df.head(3)
            for idx, row in top_3.iterrows():
                st.write(f"**{row['Failure Type']}**: {row['Probability']:.1%}")
            
            st.markdown("---")
    
    # Model insights
    st.markdown("---")
    st.subheader("🎯 Model Insights")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h3 style="margin: 0 0 1rem 0;">📋 Current Parameters</h3>
        </div>
        """, unsafe_allow_html=True)
        
        params_df = pd.DataFrame({
            'Parameter': ['Machine Type', 'Air Temperature', 'Process Temperature', 
                         'Rotational Speed', 'Torque', 'Tool Wear'],
            'Value': [machine_type, f"{air_temp} K", f"{process_temp} K", 
                     f"{rotational_speed} rpm", f"{torque} Nm", f"{tool_wear} min"]
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    with col4:
        st.markdown("""
        <div class="card">
            <h3 style="margin: 0 0 1rem 0;">⚠️ Model Notes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Model Capabilities:**
        - ✅ Tool Wear Detection
        - ✅ Power Failure Prediction
        - ✅ Normal Operation Detection
        
        **Limitations:**
        - Heat Dissipation: Limited detection
        - Overstrain: Rare predictions
        """)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if tool_wear > 250:
        st.warning("🔧 **Critical Tool Wear** - Replace immediately")
    elif tool_wear > 200:
        st.info("⚠️ **High Tool Wear** - Schedule maintenance")
    else:
        st.success("✅ **Tool Wear Normal**")
    
    if torque > 60 and rotational_speed > 2000:
        st.warning("⚡ **High Mechanical Stress** - Reduce load")
    elif air_temp > 303 or process_temp > 313:
        st.info("🌡️ **Elevated Temperatures** - Check cooling")
    else:
        st.success("📊 **Parameters Optimal**")
    
    with col4:
        st.markdown("""
        <div class="card">
            <h3 style="margin: 0 0 1rem 0;">🏆 Performance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Power Recall', 'Tool Wear Recall'],
            'Score': ['80.88%', '100%', '100%']
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🏭 Industrial AI Solutions</h3>
        <p>Developed by J Anand | SRM Institute of Science and Technology</p>
        <p style="font-size: 0.9rem; color: var(--text-muted);">Powered by XGBoost | Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()