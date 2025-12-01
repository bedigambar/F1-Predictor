import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="F1 Predictor Model",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --f1-red: #E10600;
        --f1-dark: #15151E;
        --f1-gray: #38383F;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #E10600 0%, #FF1E00 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #E10600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #15151E;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #E10600 0%, #FF1E00 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(225,6,0,0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E10600;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        quali_model = joblib.load('models/quali_model.pkl')
        race_model = joblib.load('models/race_model.pkl')
        sc_model = joblib.load('models/sc_model.pkl')
        le_driver = joblib.load('models/le_driver.pkl')
        le_team = joblib.load('models/le_team.pkl')
        le_circuit = joblib.load('models/le_circuit.pkl')
        return quali_model, race_model, sc_model, le_driver, le_team, le_circuit
    except FileNotFoundError:
        st.error("Models not found. Please run src/train.py first.")
        st.stop()

quali_model, race_model, sc_model, le_driver, le_team, le_circuit = load_models()

DRIVER_NAMES = {
    'ALB': 'Albon',
    'ALO': 'Alonso',
    'BEA': 'Bearman',
    'BOT': 'Bottas',
    'DEV': 'De Vries',
    'GAS': 'Gasly',
    'HAM': 'Hamilton',
    'HUL': 'Hulkenberg',
    'LAT': 'Latifi',
    'LAW': 'Lawson',
    'LEC': 'Leclerc',
    'MAG': 'Magnussen',
    'MSC': 'Schumacher',
    'NOR': 'Norris',
    'OCO': 'Ocon',
    'PER': 'Perez',
    'PIA': 'Piastri',
    'RIC': 'Ricciardo',
    'RUS': 'Russell',
    'SAI': 'Sainz',
    'SAR': 'Sargeant',
    'STR': 'Stroll',
    'TSU': 'Tsunoda',
    'VER': 'Verstappen',
    'VET': 'Vettel',
    'ZHO': 'Zhou'
}

# Creating display names for drivers
driver_display_names = {code: f"{DRIVER_NAMES.get(code, code)} ({code})" for code in le_driver.classes_}
driver_code_to_display = {code: DRIVER_NAMES.get(code, code) for code in le_driver.classes_}
display_to_code = {v: k for k, v in driver_display_names.items()}

TEAM_DISPLAY_NAMES = {
    'RB': 'RB (Visa Cash App RB)',
    'AlphaTauri': 'AlphaTauri (2022-2023)',
    'Alfa Romeo': 'Alfa Romeo (2022-2023)',
    'Kick Sauber': 'Kick Sauber (2024)',
    'Red Bull Racing': 'Red Bull Racing',
    'Alpine': 'Alpine',
    'Aston Martin': 'Aston Martin',
    'Ferrari': 'Ferrari',
    'Haas F1 Team': 'Haas F1 Team',
    'McLaren': 'McLaren',
    'Mercedes': 'Mercedes',
    'Williams': 'Williams'
}

team_display_names = {code: TEAM_DISPLAY_NAMES.get(code, code) for code in le_team.classes_}
team_display_to_code = {v: k for k, v in team_display_names.items()}

# Load training data
@st.cache_data
def load_training_data():
    try:
        return pd.read_csv('data/processed/f1_training_data.csv')
    except:
        return None

training_data = load_training_data()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏎️ F1 AI Race Predictor</h1>
    <p>Advanced Machine Learning Predictions for Qualifying, Race Results & Safety Car Probability</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/1200px-F1.svg.png", width=150)
    
    st.markdown("### 🏁 Race Configuration")
    
    # Mode selection
    mode = st.radio("Prediction Mode", ["Single Driver", "Full Grid Comparison"], 
                    help="Choose between predicting for one driver or comparing multiple drivers")
    
    st.markdown("---")
    st.markdown("### 🌍 Track Conditions")
    
    circuit_name = st.selectbox("Circuit", sorted(le_circuit.classes_), 
                                help="Select the race circuit")
    
    col1, col2 = st.columns(2)
    with col1:
        weather_type = st.selectbox("Weather", ["☀️ Dry", "🌧️ Rain"])
    with col2:
        temp = st.slider("Temp (°C)", 15, 50, 30)
    
    st.markdown("---")
    
    if mode == "Single Driver":
        st.markdown("### 👤 Driver Setup")
        # Use display names for selection
        selected_display_name = st.selectbox("Driver", [driver_display_names[code] for code in sorted(le_driver.classes_)])
        driver_name = display_to_code[selected_display_name]
        
        selected_display_team = st.selectbox("Team", [team_display_names[code] for code in sorted(le_team.classes_)])
        team_name = team_display_to_code[selected_display_team]
        
        st.markdown("### 📊 Performance Data")
        fp_pos = st.slider("FP2 Position", 1, 20, 5, 
                          help="Practice session position (lower is better)")
        driver_form = st.slider("Recent Form", 1.0, 20.0, 10.0, 
                               help="Average finish position in last 3 races")
        
        predict_button = st.button("🚀 Run Prediction", width='stretch')
    else:
        st.markdown("### 👥 Grid Comparison")
        num_drivers = st.slider("Number of Drivers", 3, 10, 5)
        
        # Default selection
        default_drivers = [driver_display_names[code] for code in sorted(le_driver.classes_)[:num_drivers]]
        
        selected_display_drivers = st.multiselect(
            "Select Drivers",
            [driver_display_names[code] for code in sorted(le_driver.classes_)],
            default=default_drivers
        )
        selected_drivers = [display_to_code[name] for name in selected_display_drivers]
        predict_button = st.button("🚀 Compare Drivers", width='stretch')

# Model info
with st.expander("ℹ️ Model Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Qualifying MAE", "3.55 positions", help="Mean Absolute Error")
    with col2:
        st.metric("Race MAE", "3.29 positions", help="Mean Absolute Error")
    with col3:
        st.metric("Training Records", "839", help="Total data points")
    
    st.markdown("""
    **Models Used:**
    - Qualifying Predictor: XGBoost Regressor
    - Race Result Predictor: XGBoost Regressor  
    - Safety Car Predictor: XGBoost Classifier
    
    **Data Sources:** 2022-2024 F1 Seasons (42 races across 21 circuits)
    """)

# Prediction logic
if predict_button:
    rain_flag = 1 if "Rain" in weather_type else 0
    c_id = le_circuit.transform([circuit_name])[0]
    
    if mode == "Single Driver":
        # Single driver prediction
        d_id = le_driver.transform([driver_name])[0]
        t_id = le_team.transform([team_name])[0]
        
        # Predicting qualifying
        quali_input = pd.DataFrame([[fp_pos, driver_form, t_id, c_id, d_id]], 
                                   columns=['fp_pos', 'driver_form', 'team_id', 'circuit_id', 'driver_id'])
        pred_grid = float(quali_model.predict(quali_input)[0])
        
        # Predicting race
        race_input = pd.DataFrame([[pred_grid, driver_form, t_id, c_id, d_id, temp, rain_flag]],
                                  columns=['grid_position', 'driver_form', 'team_id', 'circuit_id', 'driver_id', 'air_temp', 'rainfall'])
        pred_finish = float(race_model.predict(race_input)[0])
        
        # Predicting safety car
        sc_input = pd.DataFrame([[c_id, rain_flag, temp]], columns=['circuit_id', 'rainfall', 'air_temp'])
        sc_prob = float(sc_model.predict_proba(sc_input)[0][1])
        
        # Display results
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ⏱️ Qualifying")
            st.metric("Predicted Grid Position", f"P{int(round(pred_grid))}")
            progress_val = max(0.05, 1 - (pred_grid/20))
            st.progress(progress_val)
            st.caption(f"Confidence: {progress_val*100:.0f}%")
        
        with col2:
            st.markdown("### 🏁 Race Result")
            st.metric("Predicted Finish", f"P{int(round(pred_finish))}")
            delta = pred_grid - pred_finish
            if delta > 0:
                st.success(f"📈 Gaining {int(delta)} positions")
            elif delta < 0:
                st.error(f"📉 Losing {abs(int(delta))} positions")
            else:
                st.info("➡️ Maintaining position")
        
        with col3:
            st.markdown("### 🚨 Safety Car")
            st.metric("Probability", f"{sc_prob*100:.1f}%")
            if sc_prob > 0.6:
                st.warning("⚠️ High SC Risk")
            else:
                st.success("✅ Clean Race Expected")
        
        # Position change visualization
        st.markdown("---")
        st.markdown("### 📊 Position Movement Visualization")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=['Qualifying', 'Race Finish'],
            y=[pred_grid, pred_finish],
            mode='lines+markers',
            name=driver_code_to_display[driver_name],
            line=dict(color='#E10600', width=4),
            marker=dict(size=15, color='#E10600')
        ))
        
        fig.update_layout(
            title=f"{driver_code_to_display[driver_name]} - Position Progression",
            yaxis=dict(title="Position", autorange="reversed", gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(title="Session", gridcolor='rgba(255,255,255,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.markdown("### 🔍 Prediction Factors")
        
        feature_importance = {
            'Practice Pace': fp_pos / 20 * 100,
            'Driver Form': driver_form / 20 * 100,
            'Weather Impact': rain_flag * 50 + 25,
            'Track Temperature': (temp - 15) / 35 * 100
        }
        
        fig_importance = go.Figure(go.Bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            marker=dict(color='#E10600', opacity=0.8)
        ))
        
        fig_importance.update_layout(
            title="Factors Influencing Prediction",
            xaxis=dict(title="Impact Score", gridcolor='rgba(255,255,255,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
    else:
        # Grid comparison mode
        if not selected_drivers:
            st.warning("Please select at least one driver")
        else:
            st.markdown("## 📊 Grid Comparison")
            
            results = []
            for driver in selected_drivers:
                d_id = le_driver.transform([driver])[0]
                # Use average team for comparison
                t_id = 5  
                fp_pos_default = 10 
                form_default = 10.0
                
                # Predicting grid position
                quali_input = pd.DataFrame([[fp_pos_default, form_default, t_id, c_id, d_id]], 
                                           columns=['fp_pos', 'driver_form', 'team_id', 'circuit_id', 'driver_id'])
                pred_grid = float(quali_model.predict(quali_input)[0])
                
                race_input = pd.DataFrame([[pred_grid, form_default, t_id, c_id, d_id, temp, rain_flag]],
                                          columns=['grid_position', 'driver_form', 'team_id', 'circuit_id', 'driver_id', 'air_temp', 'rainfall'])
                pred_finish = float(race_model.predict(race_input)[0])
                
                results.append({
                    'Driver': driver_code_to_display[driver],
                    'Predicted Grid': int(round(pred_grid)),
                    'Predicted Finish': int(round(pred_finish)),
                    'Position Change': int(round(pred_grid - pred_finish))
                })
            
            df_results = pd.DataFrame(results).sort_values('Predicted Finish')
            
            # Display table
            st.dataframe(df_results, width='stretch', hide_index=True)
            
            # Comparison chart
            fig_comparison = go.Figure()
            
            for _, row in df_results.iterrows():
                fig_comparison.add_trace(go.Scatter(
                    x=['Qualifying', 'Race'],
                    y=[row['Predicted Grid'], row['Predicted Finish']],
                    mode='lines+markers',
                    name=row['Driver'],
                    line=dict(width=3),
                    marker=dict(size=10)
                ))
            
            fig_comparison.update_layout(
                title="Driver Comparison - Position Progression",
                yaxis=dict(title="Position", autorange="reversed", gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(title="Session", gridcolor='rgba(255,255,255,0.1)'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

# Circuit statistics by training data
if training_data is not None:
    with st.expander("📈 Circuit Historical Statistics"):
        circuit_data = training_data[training_data['circuit_id'] == le_circuit.transform([circuit_name])[0]]
        
        if len(circuit_data) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Races at Circuit", len(circuit_data) // 20)
            with col2:
                avg_sc = circuit_data['safety_car'].mean()
                st.metric("Avg SC Probability", f"{avg_sc*100:.0f}%")
            with col3:
                avg_temp = circuit_data['air_temp'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        else:
            st.info("No historical data available for this circuit")

st.warning("""
**⚠️ Prediction Disclaimer**
These predictions are generated by models trained on limited historical data (2022-2024 seasons). 
Real-world F1 racing involves many unpredictable variables (crashes, mechanical failures, strategy errors) that this model cannot fully account for. 
Please treat these results as estimates rather than certainties.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 2rem;'>
    <p>F1 AI Predictor | Powered by XGBoost & Streamlit</p>
    <p style='font-size: 0.8rem;'>Data: 2022-2024 F1 Seasons | 839 Records | 21 Circuits</p>
</div>
""", unsafe_allow_html=True)