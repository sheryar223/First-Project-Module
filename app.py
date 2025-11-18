"""
Calorie Burn Predictor - Complete Streamlit Application
Authors: Sheryar & Shamoon Waheed
UTF-8 Encoded, Production-Ready Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide"
)

# Paths
MODEL_DIR = Path('models')

# Workout configurations (FIXED: Realistic time_per_rep values)
WORKOUT_CONFIG = {
    'Pushups': {'type': 'sets', 'time_per_rep': 10, 'default_sets': 3, 'default_reps': 15, 'max_sets': 10, 'max_reps': 50, 'icon': '💪'},
    'Pullups': {'type': 'sets', 'time_per_rep': 12, 'default_sets': 3, 'default_reps': 10, 'max_sets': 8, 'max_reps': 30, 'icon': '🤸'},
    'Burpees': {'type': 'sets', 'time_per_rep': 10, 'default_sets': 3, 'default_reps': 12, 'max_sets': 8, 'max_reps': 30, 'icon': '🔥'},
    'Cycling': {'type': 'time', 'default_duration': 30, 'icon': '🚴'},
    'Running_in_Place': {'type': 'time', 'default_duration': 25, 'icon': '🏃'},
    'Walking': {'type': 'time', 'default_duration': 35, 'icon': '🚶'},
    'Jumping_Jacks': {'type': 'time', 'default_duration': 20, 'icon': '🤾'},
    'Hill_Up': {'type': 'time', 'default_duration': 30, 'icon': '⛰️'},
    'Hill_Down': {'type': 'time', 'default_duration': 25, 'icon': '⛰️'},
    'Hill_Straight': {'type': 'time', 'default_duration': 30, 'icon': '🏔️'},
    'Yoga': {'type': 'time', 'default_duration': 40, 'icon': '🧘'}  # Kept - it's in training data
}

# Health conditions
HEALTH_CONDITIONS = {
    'None': {'factor': 1.0, 'description': 'No conditions affecting metabolism', 'category': 'normal'},
    'Hypothyroidism': {'factor': 0.85, 'description': 'Underactive thyroid - reduces metabolism by ~15%', 'category': 'endocrine'},
    'Hyperthyroidism': {'factor': 1.20, 'description': 'Overactive thyroid - increases metabolism by ~20%', 'category': 'endocrine'},
    'Type 2 Diabetes': {'factor': 0.90, 'description': 'May reduce calorie burn efficiency by ~10%', 'category': 'metabolic'},
    'PCOS': {'factor': 0.88, 'description': 'Polycystic Ovary Syndrome - slows metabolism by ~12%', 'category': 'endocrine'},
    'Cushing\'s Syndrome': {'factor': 0.80, 'description': 'High cortisol - reduces metabolism by ~20%', 'category': 'endocrine'},
    'Heart Disease': {'factor': 0.85, 'description': 'Reduced cardiovascular efficiency', 'category': 'cardiovascular'},
    'Hypertension (Controlled)': {'factor': 0.95, 'description': 'Minimal impact if well-controlled', 'category': 'cardiovascular'},
    'Asthma': {'factor': 0.92, 'description': 'May limit exercise intensity', 'category': 'respiratory'},
    'COPD': {'factor': 0.80, 'description': 'Chronic lung disease - reduces exercise capacity', 'category': 'respiratory'},
    'Chronic Fatigue Syndrome': {'factor': 0.75, 'description': 'Significantly reduces exercise capacity', 'category': 'chronic'},
    'Fibromyalgia': {'factor': 0.85, 'description': 'Chronic pain may reduce exercise intensity', 'category': 'chronic'},
    'Anemia': {'factor': 0.88, 'description': 'Low oxygen delivery reduces efficiency', 'category': 'blood'},
    'Obesity (BMI>35)': {'factor': 1.15, 'description': 'Higher energy expenditure for same activity', 'category': 'metabolic'},
    'Beta-blockers': {'factor': 0.92, 'description': 'Reduces heart rate response', 'category': 'medication'},
    'Antidepressants (SSRI)': {'factor': 0.93, 'description': 'May slow metabolism slightly', 'category': 'medication'},
    'Corticosteroids': {'factor': 0.88, 'description': 'Long-term use can affect metabolism', 'category': 'medication'},
    'Other/Custom': {'factor': 1.0, 'description': 'Set custom adjustment factor', 'category': 'custom'}
}

# Helper functions
def sets_to_duration(sets, reps, time_per_rep):
    """Convert sets/reps to duration in minutes"""
    total_seconds = sets * reps * time_per_rep
    total_seconds += (sets - 1) * 60  # Rest between sets
    return total_seconds / 60

def export_session_data():
    """Export current session to JSON"""
    session_data = {
        'user_name': st.session_state.user_name if 'user_name' in st.session_state else '',
        'profile': st.session_state.user_profile if 'user_profile' in st.session_state else {},
        'history': st.session_state.history if 'history' in st.session_state else [],
        'exported_at': datetime.now().isoformat()
    }
    return json.dumps(session_data, indent=2, default=str)

def import_session_data(json_str):
    """Import session from JSON with proper type conversion"""
    try:
        data = json.loads(json_str)
        if 'user_name' in data:
            st.session_state.user_name = data['user_name']
        if 'profile' in data:
            st.session_state.user_profile = data['profile']
            st.session_state.profile_locked = True
        if 'history' in data and len(data['history']) > 0:
            for item in data['history']:
                if isinstance(item['timestamp'], str):
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                item['calories'] = float(item['calories']) if item['calories'] else 0.0
                item['duration'] = float(item['duration']) if item['duration'] else 0.0
                item['heart_rate'] = int(float(item['heart_rate'])) if item['heart_rate'] else 0
            st.session_state.history = data['history']
        return True
    except Exception as e:
        st.error(f"❌ Import failed: {str(e)}")
        return False

def calculate_calories(gender, age, height, weight, duration, heart_rate, body_temp, met_value, health_factor):
    """Calculate calories with fresh prediction each time"""
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [int(age)],
        'Height': [float(height)],
        'Weight': [float(weight)],
        'Duration': [float(duration)],
        'Heart_Rate': [int(heart_rate)],
        'Body_Temp': [float(body_temp)],
        'MET': [float(met_value)]
    })
    
    X_input = preprocessor.transform(input_data)
    prediction_raw = model.predict(X_input)[0]
    
    CORRECTION_FACTOR = 1.30
    prediction = prediction_raw * CORRECTION_FACTOR * health_factor
    
    return float(prediction)

@st.cache_resource
def load_model():
    """Load model, preprocessor, and MET mapping"""
    try:
        model = joblib.load(MODEL_DIR / 'calories_model.pkl')
        preprocessor = joblib.load(MODEL_DIR / 'preprocessor.pkl')
        met_mapping = joblib.load(MODEL_DIR / 'met_mapping.pkl')
        
        with open(MODEL_DIR / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, preprocessor, met_mapping, metadata
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please run: python scripts/train_model.py")
        st.stop()

# Load model
model, preprocessor, met_mapping, metadata = load_model()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'profile_locked' not in st.session_state:
    st.session_state.profile_locked = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'gender': 'male',
        'age': 25,
        'height': 170,
        'weight': 70,
        'health_condition': 'None'
    }

# User name screen
if st.session_state.user_name is None:
    st.title("🔥 Calorie Burn Predictor")
    st.markdown("### 👋 Welcome! Let's get started")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_name_input = st.text_input("What's your name?", placeholder="Enter your name")
    with col2:
        st.write("")
        st.write("")
        if st.button("🚀 Start", type="primary", disabled=not user_name_input):
            st.session_state.user_name = user_name_input
            st.rerun()
    
    st.info("💡 Your data stays in your browser session")
    
    st.markdown("---")
    st.markdown("**Or load previous session:**")
    uploaded_file = st.file_uploader("📂 Upload (.json)", type=['json'])
    if uploaded_file is not None:
        json_str = uploaded_file.read().decode('utf-8')
        if import_session_data(json_str):
            st.success("✅ Loaded!")
            st.rerun()
    
    st.stop()

# Header
st.title(f"🔥 Calorie Burn Predictor - Welcome {st.session_state.user_name}!")
st.markdown(f"**AI-powered** • R²={metadata['r2_test']:.4f} • Error: ±{metadata['mae_test']:.1f} kcal")
st.markdown("---")

# Sidebar - Profile
st.sidebar.header("👤 Your Profile")

if st.session_state.profile_locked:
    # LOCKED STATE
    st.sidebar.success("🔒 Profile Locked")
    
    st.sidebar.info(f"""
    **Current Profile:**
    - **Name:** {st.session_state.user_name}
    - **Gender:** {st.session_state.user_profile['gender'].title()}
    - **Age:** {st.session_state.user_profile['age']} years
    - **Weight:** {st.session_state.user_profile['weight']} kg
    - **Height:** {st.session_state.user_profile['height']} cm
    - **Health:** {st.session_state.user_profile['health_condition']}
    """)
    
    gender = st.session_state.user_profile['gender']
    age = st.session_state.user_profile['age']
    height = st.session_state.user_profile['height']
    weight = st.session_state.user_profile['weight']
    health_condition = st.session_state.user_profile['health_condition']
    
    if health_condition == 'Other/Custom':
        health_factor = st.session_state.user_profile.get('health_factor', 1.0)
    else:
        health_factor = HEALTH_CONDITIONS[health_condition]['factor']
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔓 Edit Profile**")
    
    unlock_checkbox = st.sidebar.checkbox(
        "⚠️ This will clear your workout history. Continue?",
        key="unlock_checkbox"
    )
    
    if st.sidebar.button("✅ Unlock & Clear History", type="secondary", disabled=not unlock_checkbox):
        st.session_state.profile_locked = False
        st.session_state.history = []
        st.success("✅ Profile unlocked! History cleared.")
        st.rerun()

else:
    # UNLOCKED STATE
    st.sidebar.warning("⚠️ Lock profile to start tracking")
    
    gender = st.sidebar.selectbox("Gender", ["male", "female"])
    age = st.sidebar.number_input("Age (years)", 16, 80, st.session_state.user_profile['age'])
    height = st.sidebar.number_input("Height (cm)", 140, 220, st.session_state.user_profile['height'])
    weight = st.sidebar.number_input("Weight (kg)", 40, 150, st.session_state.user_profile['weight'])
    
    st.sidebar.markdown("---")
    st.sidebar.header("🏥 Health Status")
    health_condition = st.sidebar.selectbox(
        "Medical conditions?",
        options=list(HEALTH_CONDITIONS.keys())
    )
    
    if health_condition == 'Other/Custom':
        custom_adjustment = st.sidebar.slider("Metabolism %", 50, 150, 100, 5)
        health_factor = custom_adjustment / 100
    else:
        health_factor = HEALTH_CONDITIONS[health_condition]['factor']
        if health_condition != 'None':
            st.sidebar.info(f"ℹ️ {HEALTH_CONDITIONS[health_condition]['description']}")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🔒 Lock Profile & Start Tracking", type="primary"):
        st.session_state.profile_locked = True
        st.session_state.user_profile = {
            'gender': gender,
            'age': age,
            'height': height,
            'weight': weight,
            'health_condition': health_condition,
            'health_factor': health_factor if health_condition == 'Other/Custom' else None
        }
        st.success("✅ Profile locked!")
        st.rerun()

if not st.session_state.profile_locked:
    st.warning("👈 Please lock your profile in the sidebar to begin")
    st.stop()

# Workout Session
st.sidebar.markdown("---")
st.sidebar.header("🏃 Workout")

workout = st.sidebar.selectbox(
    "Type",
    options=list(WORKOUT_CONFIG.keys()),
    format_func=lambda x: f"{WORKOUT_CONFIG[x]['icon']} {x.replace('_', ' ')}"
)

workout_info = WORKOUT_CONFIG[workout]

if workout_info['type'] == 'sets':
    st.sidebar.markdown(f"**{workout_info['icon']} Sets & Reps**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sets = st.number_input("Sets", 1, workout_info['max_sets'], workout_info['default_sets'])
    with col2:
        reps = st.number_input("Reps", 1, workout_info['max_reps'], workout_info['default_reps'])
    
    duration = sets_to_duration(sets, reps, workout_info['time_per_rep'])
    st.sidebar.caption(f"⏱️ ~{duration:.1f} min")
    workout_display = f"{sets}×{reps}"
else:
    duration = st.sidebar.slider("Duration (min)", 5, 120, workout_info['default_duration'])
    workout_display = f"{duration} min"

st.sidebar.markdown("---")
heart_rate = st.sidebar.slider("Heart Rate (BPM)", 60, 180, 100)
st.sidebar.caption("💡 Measure: Count pulse for 15 sec × 4")

# Heart Rate Guide
with st.sidebar.expander("📖 Heart Rate Guide"):
    st.markdown("""
    **Typical HR Zones:**
    
    **🚶 Light (60-100 BPM)**
    - Walking, Yoga, Stretching
    - Fat burning zone
    
    **🚴 Moderate (100-130 BPM)**
    - Brisk walking, Cycling
    - Cardio fitness zone
    
    **🏃 Vigorous (130-150 BPM)**
    - Running, Swimming
    - Aerobic zone
    
    **🔥 Intense (150-170 BPM)**
    - HIIT, Burpees, Sprints
    - Anaerobic zone
    
    **💡 How to measure:**
    1. Find pulse (wrist or neck)
    2. Count beats for 15 seconds
    3. Multiply by 4 = Your BPM
    
    **⚠️ Max HR = 220 - your age**
    Your max: ~{220 - age} BPM
    """)

body_temp = st.sidebar.slider("Body Temp (°C)", 36.0, 40.0, 37.2, 0.1)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔥 Predict", type="primary", width='stretch')

# Data Management
st.sidebar.markdown("---")
st.sidebar.header("💾 Data")

col_save, col_load = st.sidebar.columns(2)

with col_save:
    if st.session_state.profile_locked:
        if st.button("💾 Save", key="save_btn_sidebar"):
            session_json = export_session_data()
            st.download_button(
                label="📥 Download",
                data=session_json,
                file_name=f"session_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key=f"download_session_{datetime.now().timestamp()}"
            )

with col_load:
    if st.button("👋 Logout", key="logout_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📅 Plans", "📈 History"])

# TAB 1: PREDICTION
with tab1:
    if predict_btn:
        try:
            prediction = calculate_calories(
                gender, age, height, weight, duration,
                heart_rate, body_temp, met_mapping[workout], health_factor
            )
            
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calories", f"{prediction:.1f} kcal")
            with col2:
                lower = prediction - metadata['mae_test']
                upper = prediction + metadata['mae_test']
                st.metric("Range", f"{lower:.0f}-{upper:.0f}")
            with col3:
                st.metric("MET", f"{met_mapping[workout]:.1f}")
            with col4:
                rate = prediction / duration
                st.metric("Rate", f"{rate:.1f} kcal/min")
            
            if health_condition != 'None':
                impact = (health_factor - 1) * 100
                st.warning(f"⚕️ {health_condition} adjusted by {impact:+.0f}%")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Session:**
                - Workout: {workout_display}
                - Duration: {duration:.1f} min
                - Burn: {prediction:.1f} kcal
                """)
            with col2:
                intensity = "Low" if met_mapping[workout] < 4 else "Moderate" if met_mapping[workout] < 7 else "High"
                st.success(f"""
                **Quality:**
                - Intensity: {intensity}
                - Rate: {rate:.1f} kcal/min
                - For {weight}kg
                """)
            
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'workout': workout,
                'workout_display': workout_display,
                'duration': duration,
                'calories': prediction,
                'heart_rate': heart_rate,
                'health_condition': health_condition
            })
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("👈 Enter details and click **Predict**")
        
        # ONLY show usage guide - NO PLANS HERE!
        with st.expander("📖 How to Use"):
            st.markdown("""
            **Step 1:** Fill profile and lock it
            **Step 2:** Select workout type
            **Step 3:** Enter duration or sets/reps
            **Step 4:** Measure heart rate
            **Step 5:** Click Predict
            """)

# TAB 2: PLANS
with tab2:
    st.subheader("🎯 Personalized Workout Plans")
    
    # Get current profile
    current_gender = st.session_state.user_profile['gender']
    current_age = st.session_state.user_profile['age']
    current_height = st.session_state.user_profile['height']
    current_weight = st.session_state.user_profile['weight']
    current_health = st.session_state.user_profile['health_condition']
    
    if current_health == 'Other/Custom':
        current_health_factor = st.session_state.user_profile.get('health_factor', 1.0)
    else:
        current_health_factor = HEALTH_CONDITIONS[current_health]['factor']
    
    # Calculate BMI
    bmi = current_weight / ((current_height/100) ** 2)
    
    # Determine fitness category
    if current_age < 30 and bmi < 25:
        default_intensity = 'High'
        fitness_category = "Young & Fit"
    elif current_age < 50 and bmi < 28:
        default_intensity = 'Moderate'
        fitness_category = "Average Fitness"
    else:
        default_intensity = 'Low'
        fitness_category = "Beginner/Senior"
    
    # BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    # Show suggestion
    st.info(f"""
    **🎯 Workout Plan Suggestion:**
    
    Based on your profile:
    - **Age:** {current_age} years → *{fitness_category}*
    - **BMI:** {bmi:.1f} → *{bmi_category}*
    - **Health:** {current_health}
    
    **We recommend:** **{default_intensity} Intensity** ✅
    """)
    
    # Intensity selector
    intensity = st.selectbox(
        "Select Workout Intensity Level",
        ['Low', 'Moderate', 'High'],
        index=['Low', 'Moderate', 'High'].index(default_intensity)
    )
    
    # Plans dictionary
    plans = {
        'Low': {
            'name': 'Beginner Plan',
            'workouts': [
                {'day': 1, 'workout': 'Walking', 'value': 25},
                {'day': 2, 'workout': 'Yoga', 'value': 30},
                {'day': 3, 'workout': 'Walking', 'value': 30},
                {'day': 4, 'workout': 'Pushups', 'value': {'sets': 2, 'reps': 8}},
                {'day': 5, 'workout': 'Yoga', 'value': 35},
                {'day': 6, 'workout': 'Cycling', 'value': 20},
                {'day': 7, 'workout': 'Rest', 'value': 0}
            ]
        },
        'Moderate': {
            'name': 'Intermediate Plan',
            'workouts': [
                {'day': 1, 'workout': 'Cycling', 'value': 30},
                {'day': 2, 'workout': 'Pushups', 'value': {'sets': 3, 'reps': 15}},
                {'day': 3, 'workout': 'Walking', 'value': 35},
                {'day': 4, 'workout': 'Jumping_Jacks', 'value': 20},
                {'day': 5, 'workout': 'Yoga', 'value': 30},
                {'day': 6, 'workout': 'Burpees', 'value': {'sets': 2, 'reps': 10}},
                {'day': 7, 'workout': 'Rest', 'value': 0}
            ]
        },
        'High': {
            'name': 'Advanced Plan',
            'workouts': [
                {'day': 1, 'workout': 'Running_in_Place', 'value': 45},
                {'day': 2, 'workout': 'Burpees', 'value': {'sets': 5, 'reps': 15}},
                {'day': 3, 'workout': 'Hill_Up', 'value': 40},
                {'day': 4, 'workout': 'Pullups', 'value': {'sets': 4, 'reps': 12}},
                {'day': 5, 'workout': 'Cycling', 'value': 60},
                {'day': 6, 'workout': 'Pushups', 'value': {'sets': 5, 'reps': 20}},
                {'day': 7, 'workout': 'Walking', 'value': 30}
            ]
        }
    }
    
    # Health restrictions
    if current_health in ['Heart Disease', 'COPD', 'Chronic Fatigue Syndrome'] and intensity == 'High':
        st.error(f"⚠️ HIGH intensity NOT recommended for {current_health}!")
        st.warning("🔽 Auto-adjusting to MODERATE")
        intensity = 'Moderate'
    
    selected_plan = plans[intensity]
    st.markdown(f"### 📅 {selected_plan['name']}")
    
    # Calculate calories using ML model
    total_burn = 0
    plan_data = []
    
    plan_hr = 110
    plan_temp = 37.3
    
    for item in selected_plan['workouts']:
        if item['workout'] != 'Rest':
            workout_type = item['workout']
            workout_cfg = WORKOUT_CONFIG[workout_type]
            
            if workout_cfg['type'] == 'sets':
                s = item['value']['sets']
                r = item['value']['reps']
                dur = sets_to_duration(s, r, workout_cfg['time_per_rep'])
                display_val = f"{s}×{r}"
            else:
                dur = item['value']
                display_val = f"{dur} min"
            
            # Use ML model
            est_cal = calculate_calories(
                current_gender,
                current_age,
                current_height,
                current_weight,
                dur,
                plan_hr,
                plan_temp,
                met_mapping[workout_type],
                current_health_factor
            )
            
            total_burn += est_cal
            
            plan_data.append({
                'Day': f"Day {item['day']}",
                'Workout': f"{workout_cfg['icon']} {workout_type.replace('_', ' ')}",
                'Volume': display_val,
                'Calories': f"{est_cal:.0f} kcal"
            })
        else:
            plan_data.append({
                'Day': f"Day {item['day']}",
                'Workout': "🛌 Rest",
                'Volume': "-",
                'Calories': "-"
            })
    
    df_plan = pd.DataFrame(plan_data)
    st.dataframe(df_plan, width='stretch', hide_index=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Weekly", f"{total_burn:.0f} kcal")
    with col2:
        st.metric("Daily Avg", f"{total_burn/7:.0f} kcal")
    with col3:
        active = len([p for p in plan_data if p['Workout'] != "🛌 Rest"])
        st.metric("Active Days", f"{active}/7")
    with col4:
        loss = total_burn / 7700
        st.metric("Est. Loss", f"{loss:.2f} kg/week")
    
    st.markdown("---")
    
    plan_header = f"""Plan for {st.session_state.user_name}
Date: {datetime.now().strftime('%Y-%m-%d')}
Weight: {current_weight}kg | BMI: {bmi:.1f}
Intensity: {intensity}

"""
    full_csv = plan_header + df_plan.to_csv(index=False)
    
    st.download_button(
        label=f"📥 Download Plan",
        data=full_csv.encode('utf-8'),
        file_name=f"plan_{st.session_state.user_name}_{intensity}.csv",
        mime="text/csv",
        key=f"plan_{int(datetime.now().timestamp())}"
    )

# TAB 3: HISTORY
with tab3:
    st.subheader(f"📈 History - {st.session_state.user_name}")
    
    st.info(f"""**Tracking:** {gender.title()} • {age}y • {weight}kg • {health_condition}""")
    
    if st.session_state.history and len(st.session_state.history) > 0:
        df_history = pd.DataFrame(st.session_state.history)
        
        df_history['calories'] = pd.to_numeric(df_history['calories'], errors='coerce').fillna(0.0)
        df_history['duration'] = pd.to_numeric(df_history['duration'], errors='coerce').fillna(0.0)
        df_history['heart_rate'] = pd.to_numeric(df_history['heart_rate'], errors='coerce').fillna(0).astype(int)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sessions", len(df_history))
        with col2:
            st.metric("Total", f"{df_history['calories'].sum():.0f} kcal")
        with col3:
            st.metric("Avg Duration", f"{df_history['duration'].mean():.0f} min")
        with col4:
            st.metric("Avg HR", f"{df_history['heart_rate'].mean():.0f} BPM")
        
        st.markdown("---")
        
        display_df = df_history.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df[['timestamp', 'workout', 'workout_display', 'calories', 'heart_rate', 'health_condition']]
        display_df.columns = ['Time', 'Workout', 'Volume', 'Calories (kcal)', 'HR (BPM)', 'Health']
        display_df['Calories (kcal)'] = display_df['Calories (kcal)'].apply(lambda x: f"{x:.0f}")
        display_df['HR (BPM)'] = display_df['HR (BPM)'].apply(lambda x: f"{x}")
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        if len(df_history) > 1:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Calories by Workout**")
                workout_sum = df_history.groupby('workout')['calories'].sum()
                st.bar_chart(workout_sum)
            
            with col2:
                st.markdown("**Calories Over Time**")
                plot_df = df_history.copy()
                if not pd.api.types.is_datetime64_any_dtype(plot_df['timestamp']):
                    plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
                time_data = plot_df.set_index('timestamp')['calories']
                st.line_chart(time_data)
        
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            csv_export = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History",
                data=csv_export,
                file_name=f"history_{st.session_state.user_name}.csv",
                mime="text/csv",
                key=f"export_{int(datetime.now().timestamp())}"
            )
        
        with col_btn2:
            if st.button("🗑️ Clear History"):
                if st.checkbox("⚠️ Confirm?"):
                    st.session_state.history = []
                    st.success("✅ Cleared!")
                    st.rerun()
    else:
        st.info("📭 No history yet!")
# Footer
st.markdown("---")
col_footer1, col_footer2 = st.columns([3, 1])

with col_footer1:
    st.caption("🤖 Built with XGBoost & Streamlit | FYP Project by Sheryar & Shamoon Waheed")

with col_footer2:
    if st.session_state.history:
        total_burned = sum([h['calories'] for h in st.session_state.history])
        st.caption(f"💪 Total: {total_burned:.0f} kcal burned")