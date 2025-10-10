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

# Modern Minimalist CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Source+Code+Pro:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --bg-card: #0d1117;
        --bg-elevated: #21262d;
        --border-color: #30363d;
        --border-subtle: #21262d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #484f58;
        --accent-teal: #14b8a6;
        --accent-orange: #fb923c;
        --accent-green: #22c55e;
        --accent-red: #ef4444;
        --accent-blue: #3b82f6;
        --gradient-teal: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        --gradient-orange: linear-gradient(135deg, #fb923c 0%, #f97316 100%);
        --gradient-success: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        --gradient-danger: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        --gradient-card: linear-gradient(145deg, #0d1117 0%, #161b22 100%);
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.2);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.3);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.4);
        --shadow-xl: 0 20px 25px rgba(0,0,0,0.5);
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
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    /* Header */
    .main-header {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--gradient-teal);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: var(--text-primary);
        text-align: center;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
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
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: var(--shadow-md);
        border-color: var(--accent-teal);
    }
    
    /* Status Indicators */
    .status-normal {
        background: var(--gradient-success);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1rem;
        box-shadow: var(--shadow-md);
    }
    
    .status-warning {
        background: var(--gradient-orange);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1rem;
        box-shadow: var(--shadow-md);
    }
    
    .status-critical {
        background: var(--gradient-danger);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        font-size: 1rem;
        box-shadow: var(--shadow-lg);
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-teal);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Input Elements */
    .stSelectbox > div > div {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        color: var(--text-primary);
    }
    
    .stSlider > div > div > div {
        background: var(--accent-teal);
    }
    
    /* Prediction Card */
    .prediction-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: var(--gradient-teal);
    }
    
    /* DataFrames */
    .dataframe {
        background: var(--bg-secondary);
        border-radius: 6px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }
    
    .dataframe th {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .dataframe td {
        color: var(--text-secondary);
        border-top: 1px solid var(--border-subtle);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 6px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-teal);
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
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: var(--shadow-md);
    }
    
    .footer h3 {
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
    }
    
    .footer p {
        color: var(--text-secondary);
        margin: 0.3rem 0;
    }
    
    /* Custom Components */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent-teal);
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .status-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .status-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.8rem;
        display: flex;
        align-items: center;
        box-shadow: var(--shadow-sm);
    }
    
    .status-icon {
        font-size: 1.2rem;
        margin-right: 0.8rem;
    }
    
    .status-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
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
    <div style="text-align: center; padding: 1.5rem; background: var(--bg-tertiary); border-radius: 8px; margin-bottom: 1.5rem; border: 1px solid var(--border-color);">
        <h2 style="color: var(--text-primary); margin: 0; font-size: 1.2rem;">⚙️ Control Panel</h2>
        <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.9rem;">System Parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Scenarios
    st.sidebar.markdown("""
    <div style="background: var(--bg-tertiary); border-left: 3px solid var(--accent-teal); padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
        <h3 style="color: var(--accent-teal); margin: 0 0 0.5rem 0; font-size: 1rem;">🎯 Quick Scenarios</h3>
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
            'air_temp': '#14b8a6',
            'process_temp': '#3b82f6', 
            'rpm': '#22c55e',
            'torque': '#fb923c',
            'tool_wear': '#14b8a6',
            'status': '#22c55e'
        }
        
        # Add gauges
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=air_temp,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [295, 305], 'tickcolor': '#8b949e'},
                'bar': {'color': gauge_colors['air_temp']},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [295, 300], 'color': 'rgba(20, 184, 166, 0.1)'},
                    {'range': [300, 305], 'color': 'rgba(251, 146, 60, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#14b8a6", 'width': 2},
                    'thickness': 0.5,
                    'value': 302
                }
            }), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=process_temp,
            gauge={
                'axis': {'range': [305, 315], 'tickcolor': '#8b949e'},
                'bar': {'color': gauge_colors['process_temp']},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [305, 310], 'color': 'rgba(34, 197, 94, 0.1)'},
                    {'range': [310, 315], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#3b82f6", 'width': 2},
                    'thickness': 0.5,
                    'value': 312
                }
            }), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=rotational_speed,
            gauge={
                'axis': {'range': [1000, 3000], 'tickcolor': '#8b949e'},
                'bar': {'color': gauge_colors['rpm']},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [1000, 2000], 'color': 'rgba(34, 197, 94, 0.1)'},
                    {'range': [2000, 3000], 'color': 'rgba(251, 146, 60, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#22c55e", 'width': 2},
                    'thickness': 0.5,
                    'value': 2500
                }
            }), row=1, col=3)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=torque,
            gauge={
                'axis': {'range': [10, 80], 'tickcolor': '#8b949e'},
                'bar': {'color': gauge_colors['torque']},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [10, 50], 'color': 'rgba(34, 197, 94, 0.1)'},
                    {'range': [50, 80], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#fb923c", 'width': 2},
                    'thickness': 0.5,
                    'value': 60
                }
            }), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=tool_wear,
            gauge={
                'axis': {'range': [0, 300], 'tickcolor': '#8b949e'},
                'bar': {'color': gauge_colors['tool_wear']},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [0, 150], 'color': 'rgba(34, 197, 94, 0.1)'},
                    {'range': [150, 250], 'color': 'rgba(251, 146, 60, 0.1)'},
                    {'range': [250, 300], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#14b8a6", 'width': 2},
                    'thickness': 0.5,
                    'value': 250
                }
            }), row=2, col=2)
        
        # Status indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=100,
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                'bar': {'color': '#22c55e'},
                'bgcolor': '#161b22',
                'borderwidth': 1,
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                    {'range': [50, 100], 'color': 'rgba(34, 197, 94, 0.1)'}
                ]
            }), row=2, col=3)
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font={'color': '#f0f6fc', 'family': 'Poppins'},
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Parameters summary
        st.markdown("### 📋 Current Parameters")
        
        st.markdown("""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">""" + machine_type + """</div>
                <div class="metric-label">Machine Type</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">""" + str(air_temp) + """ K</div>
                <div class="metric-label">Air Temperature</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">""" + str(process_temp) + """ K</div>
                <div class="metric-label">Process Temperature</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">""" + str(rotational_speed) + """ rpm</div>
                <div class="metric-label">Rotational Speed</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">""" + str(torque) + """ Nm</div>
                <div class="metric-label">Torque</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">""" + str(tool_wear) + """ min</div>
                <div class="metric-label">Tool Wear</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        st.markdown("### 🔍 System Status")
        
        temp_status = "🟢 Normal" if air_temp < 302 else "🟡 High" if air_temp < 304 else "🔴 Critical"
        rpm_status = "🟢 Normal" if rotational_speed < 2500 else "🟡 High"
        torque_status = "🟢 Normal" if torque < 60 else "🟡 High" if torque < 75 else "🔴 Critical"
        wear_status = "🟢 Good" if tool_wear < 150 else "🟡 Monitor" if tool_wear < 250 else "🔴 Replace"
        
        st.markdown("""
        <div class="status-grid">
            <div class="status-item">
                <div class="status-icon">🌡️</div>
                <div class="status-text">Temperature: """ + temp_status + """</div>
            </div>
            <div class="status-item">
                <div class="status-icon">⚙️</div>
                <div class="status-text">RPM: """ + rpm_status + """</div>
            </div>
            <div class="status-item">
                <div class="status-icon">💪</div>
                <div class="status-text">Torque: """ + torque_status + """</div>
            </div>
            <div class="status-item">
                <div class="status-icon">🔧</div>
                <div class="status-text">Tool Wear: """ + wear_status + """</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
                             color='Probability', color_continuous_scale='teal',
                             title="Failure Probability Distribution")
            fig_prob.update_xaxes(tickangle=45)
            fig_prob.update_layout(
                height=350, 
                showlegend=False,
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font={'color': '#f0f6fc'}
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
        <p style="font-size: 0.9rem; color: var(--text-muted);">Powered by XGBoost Model  | Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()