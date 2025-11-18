"""
Calorie Burn Prediction - Complete Model Training Script
Authors: Sheryar & Shamoon Waheed
Purpose: Train and compare multiple ML models with comprehensive analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("CALORIE BURN PREDICTION - COMPREHENSIVE MODEL TRAINING")
print("="*70)

# 1. Setup paths
DATA_PATH = Path('data/enhanced_calories.csv')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# 2. MET values for workouts
MET_MAPPING = {
    'Pushups': 3.8,
    'Pullups': 4.0,
    'Cycling': 6.8,
    'Hill_Up': 9.0,
    'Hill_Down': 5.0,
    'Hill_Straight': 7.0,
    'Jumping_Jacks': 8.0,
    'Burpees': 8.0,
    'Running_in_Place': 7.0,
    'Walking': 3.5,
    'Yoga': 2.5
}

# 3. Load data
print("\n📁 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} rows, {df.shape[1]} columns")

# 4. Add MET values
df['MET'] = df['Workout_Type'].map(MET_MAPPING)
if df['MET'].isnull().any():
    print("⚠️ Warning: Some workout types missing MET values")
    df = df.dropna(subset=['MET'])
print(f"✓ MET values added")

# 5. Prepare features
print("\n🔧 Preparing features...")
feature_cols = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 
                'Heart_Rate', 'Body_Temp', 'MET']
X = df[feature_cols]
y = df['Calories']
print(f"✓ Features: {feature_cols}")
print(f"✓ Shape: {X.shape}")

# 6. Create preprocessor
num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'MET']
cat_cols = ['Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# 8. Preprocess
print("\n⚙️ Preprocessing data...")
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)
print(f"✓ Preprocessed shape: {X_train_pre.shape}")

# 9. Train Multiple Models
print("\n🤖 Training Multiple Models...")
print("-" * 70)

models_performance = {}

# Model 1: Linear Regression (Baseline)
print("\n1️⃣ Linear Regression (Baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train_pre, y_train)
y_pred_lr_train = lr_model.predict(X_train_pre)
y_pred_lr_test = lr_model.predict(X_test_pre)

models_performance['Linear Regression'] = {
    'model': lr_model,
    'r2_train': r2_score(y_train, y_pred_lr_train),
    'r2_test': r2_score(y_test, y_pred_lr_test),
    'mae': mean_absolute_error(y_test, y_pred_lr_test),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr_test)),
    'predictions': y_pred_lr_test
}
print(f"   R² Test: {models_performance['Linear Regression']['r2_test']:.4f}")

# Model 2: Random Forest
print("\n2️⃣ Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_pre, y_train)
y_pred_rf_train = rf_model.predict(X_train_pre)
y_pred_rf_test = rf_model.predict(X_test_pre)

models_performance['Random Forest'] = {
    'model': rf_model,
    'r2_train': r2_score(y_train, y_pred_rf_train),
    'r2_test': r2_score(y_test, y_pred_rf_test),
    'mae': mean_absolute_error(y_test, y_pred_rf_test),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf_test)),
    'predictions': y_pred_rf_test
}
print(f"   R² Test: {models_performance['Random Forest']['r2_test']:.4f}")

# Model 3: XGBoost (Best)
print("\n3️⃣ XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=150,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_pre, y_train)
y_pred_xgb_train = xgb_model.predict(X_train_pre)
y_pred_xgb_test = xgb_model.predict(X_test_pre)

models_performance['XGBoost'] = {
    'model': xgb_model,
    'r2_train': r2_score(y_train, y_pred_xgb_train),
    'r2_test': r2_score(y_test, y_pred_xgb_test),
    'mae': mean_absolute_error(y_test, y_pred_xgb_test),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb_test)),
    'predictions': y_pred_xgb_test
}
print(f"   R² Test: {models_performance['XGBoost']['r2_test']:.4f}")

print("\n✓ All models trained!")

# 10. Compare Models
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)
print(f"{'Model':<20} {'R² Train':<12} {'R² Test':<12} {'MAE (kcal)':<12} {'RMSE (kcal)':<12}")
print("-" * 70)
for name, perf in models_performance.items():
    print(f"{name:<20} {perf['r2_train']:<12.4f} {perf['r2_test']:<12.4f} {perf['mae']:<12.2f} {perf['rmse']:<12.2f}")
print("="*70)

# Select best model
best_model_name = max(models_performance.keys(), key=lambda k: models_performance[k]['r2_test'])
best_model_perf = models_performance[best_model_name]
model = best_model_perf['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² Test: {best_model_perf['r2_test']:.4f}")
print(f"   MAE: {best_model_perf['mae']:.2f} kcal")
print(f"   RMSE: {best_model_perf['rmse']:.2f} kcal")

r2_train = best_model_perf['r2_train']
r2_test = best_model_perf['r2_test']
mae = best_model_perf['mae']
rmse = best_model_perf['rmse']
y_pred_test = best_model_perf['predictions']

# 11. Create Comprehensive Visualizations
print("\n📈 Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 12))

# Panel 1: Model Comparison (R²)
ax1 = plt.subplot(3, 3, 1)
model_names = list(models_performance.keys())
r2_scores = [models_performance[m]['r2_test'] for m in model_names]
colors = ['#FF6B6B' if m != best_model_name else '#51CF66' for m in model_names]
bars = ax1.bar(model_names, r2_scores, color=colors)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Comparison (R² Test Score)')
ax1.set_ylim([min(r2_scores) - 0.01, 1.0])
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{r2_scores[i]:.4f}', ha='center', va='bottom', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: MAE Comparison
ax2 = plt.subplot(3, 3, 2)
mae_scores = [models_performance[m]['mae'] for m in model_names]
bars = ax2.bar(model_names, mae_scores, color=colors)
ax2.set_ylabel('MAE (kcal)')
ax2.set_title('Model Comparison (Mean Absolute Error)')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mae_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: RMSE Comparison
ax3 = plt.subplot(3, 3, 3)
rmse_scores = [models_performance[m]['rmse'] for m in model_names]
bars = ax3.bar(model_names, rmse_scores, color=colors)
ax3.set_ylabel('RMSE (kcal)')
ax3.set_title('Model Comparison (Root Mean Square Error)')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{rmse_scores[i]:.2f}', ha='center', va='bottom', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Actual vs Predicted
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test, y_pred_test, alpha=0.5, s=20)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Calories')
ax4.set_ylabel('Predicted Calories')
ax4.set_title(f'{best_model_name}: Actual vs Predicted (R²={r2_test:.4f})')
ax4.grid(alpha=0.3)

# Panel 5: Residuals Plot
ax5 = plt.subplot(3, 3, 5)
residuals = y_test - y_pred_test
ax5.scatter(y_pred_test, residuals, alpha=0.5, s=20)
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted Calories')
ax5.set_ylabel('Residuals (Actual - Predicted)')
ax5.set_title('Residual Plot')
ax5.grid(alpha=0.3)

# Panel 6: Error Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Prediction Error (kcal)')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Error Distribution (Mean: {residuals.mean():.2f} kcal)')
ax6.grid(alpha=0.3)

# Panel 7: Calories Distribution
ax7 = plt.subplot(3, 3, 7)
ax7.hist(y, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
ax7.set_xlabel('Calories Burned (kcal)')
ax7.set_ylabel('Frequency')
ax7.set_title('Training Data Distribution')
ax7.axvline(y.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {y.mean():.1f}')
ax7.axvline(y.median(), color='g', linestyle='--', lw=2, label=f'Median: {y.median():.1f}')
ax7.legend()
ax7.grid(alpha=0.3)

# Panel 8: Calories by Workout Type
ax8 = plt.subplot(3, 3, 8)
workout_calories = df.groupby('Workout_Type')['Calories'].mean().sort_values()
ax8.barh(workout_calories.index, workout_calories.values, color='coral')
ax8.set_xlabel('Average Calories Burned')
ax8.set_title('Calories by Workout Type')
ax8.grid(axis='x', alpha=0.3)

# Panel 9: Feature Importance
ax9 = plt.subplot(3, 3, 9)
if hasattr(model, 'feature_importances_'):
    feature_names_plot = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'MET', 'Gender']
    importances_plot = model.feature_importances_
    indices = np.argsort(importances_plot)[::-1]
    
    ax9.barh([feature_names_plot[i] for i in indices], 
             [importances_plot[i] for i in indices],
             color='lightgreen')
    ax9.set_xlabel('Importance')
    ax9.set_title(f'{best_model_name} Feature Importance')
    ax9.grid(axis='x', alpha=0.3)
else:
    ax9.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type',
             ha='center', va='center', fontsize=12)
    ax9.set_title('Feature Importance')
    ax9.axis('off')

plt.tight_layout()
plt.savefig(MODEL_DIR / 'comprehensive_analysis.png', dpi=200, bbox_inches='tight')
print("  ✓ comprehensive_analysis.png saved")

# 12. Create Model Comparison Plot
print("📊 Creating model comparison plot...")
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, perf) in enumerate(models_performance.items()):
    ax = axes[idx]
    y_pred_model = perf['predictions']
    
    ax.scatter(y_test, y_pred_model, alpha=0.5, s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Calories')
    ax.set_ylabel('Predicted Calories')
    ax.set_title(f'{model_name}\nR²={perf["r2_test"]:.4f}, MAE={perf["mae"]:.2f}')
    ax.grid(alpha=0.3)
    
    if model_name == best_model_name:
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(3)

plt.tight_layout()
plt.savefig(MODEL_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ model_comparison.png saved")

# 13. Save Model and Artifacts
print("\n💾 Saving model artifacts...")

joblib.dump(model, MODEL_DIR / 'calories_model.pkl')
print("  ✓ Model saved")

joblib.dump(preprocessor, MODEL_DIR / 'preprocessor.pkl')
print("  ✓ Preprocessor saved")

joblib.dump(MET_MAPPING, MODEL_DIR / 'met_mapping.pkl')
print("  ✓ MET mapping saved")

# Save metadata
metadata = {
    'best_model': best_model_name,
    'r2_test': float(r2_test),
    'mae_test': float(mae),
    'rmse_test': float(rmse),
    'r2_train': float(r2_train),
    'features': feature_cols,
    'all_models': {name: {'r2': float(perf['r2_test']), 'mae': float(perf['mae'])} 
                   for name, perf in models_performance.items()}
}

with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("  ✓ Metadata saved")

# 14. Create Training Summary Report
print("📄 Creating training summary...")
summary_path = MODEL_DIR / 'training_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("CALORIE BURN PREDICTION - TRAINING SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Samples: {len(df)}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write(f"Features: {', '.join(feature_cols)}\n")
    f.write(f"\nCalories Statistics:\n")
    f.write(f"  Mean: {y.mean():.2f} kcal\n")
    f.write(f"  Median: {y.median():.2f} kcal\n")
    f.write(f"  Std Dev: {y.std():.2f} kcal\n")
    f.write(f"  Min: {y.min():.2f} kcal\n")
    f.write(f"  Max: {y.max():.2f} kcal\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("MODEL COMPARISON:\n")
    f.write("="*70 + "\n")
    f.write(f"{'Model':<20} {'R² Train':<12} {'R² Test':<12} {'MAE (kcal)':<12} {'RMSE (kcal)':<12}\n")
    f.write("-" * 70 + "\n")
    for name, perf in models_performance.items():
        marker = " ← SELECTED" if name == best_model_name else ""
        f.write(f"{name:<20} {perf['r2_train']:<12.4f} {perf['r2_test']:<12.4f} {perf['mae']:<12.2f} {perf['rmse']:<12.2f}{marker}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write(f"BEST MODEL: {best_model_name}\n")
    f.write("="*70 + "\n")
    f.write(f"R² Score (Test): {r2_test:.4f}\n")
    f.write(f"Mean Absolute Error: {mae:.2f} kcal\n")
    f.write(f"Root Mean Square Error: {rmse:.2f} kcal\n")
    
    if hasattr(model, 'feature_importances_'):
        f.write(f"\nFeature Importance:\n")
        feature_names_txt = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'MET', 'Gender']
        importances_txt = model.feature_importances_
        indices = np.argsort(importances_txt)[::-1]
        for i, idx in enumerate(indices, 1):
            f.write(f"  {i}. {feature_names_txt[idx]}: {importances_txt[idx]:.4f} ({importances_txt[idx]*100:.2f}%)\n")

print("  ✓ training_summary.txt saved")

# Final Summary
print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nFiles created in '{MODEL_DIR}/':")
print("  - calories_model.pkl")
print("  - preprocessor.pkl")
print("  - met_mapping.pkl")
print("  - model_metadata.json")
print("  - comprehensive_analysis.png (9 panels)")
print("  - model_comparison.png (3 models)")
print("  - training_summary.txt")

print(f"\n🎯 Best Model: {best_model_name}")
print(f"   Accuracy: R²={r2_test:.4f} (99.{int(r2_test*10000 - 9900)}%)")
print(f"   Error: ±{mae:.2f} kcal (MAE)")
print(f"\nNext step: Run Streamlit app")
print("Command: streamlit run app/streamlit_app.py")