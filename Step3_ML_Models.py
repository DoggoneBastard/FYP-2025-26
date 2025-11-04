
# ============================================================
# STEP 3: MACHINE LEARNING MODELS
# ============================================================
# Random Forest, Gradient Boosting, and Neural Network for 
# predicting optimal cryopreservation formulations

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('MSC_Cleaned_Data.csv')

# Prepare features and target
X = df[['DMSO_Numeric', 'Cooling_Rate_Numeric']].values
y = df['Viability_Numeric'].values

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data shape:")
print(f"Training: {X_train.shape}")
print(f"Testing: {X_test.shape}")
print()

# ============================================================
# MODEL 1: RANDOM FOREST (with imputation for NaN handling)
# ============================================================
print("="*60)
print("RANDOM FOREST REGRESSOR")
print("="*60)

# Impute NaN values before training
imputer_rf = SimpleImputer(strategy='mean')
X_train_rf = imputer_rf.fit_transform(X_train)
X_test_rf = imputer_rf.transform(X_test)

# Create and train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    min_samples_split=5,   # Minimum samples to split a node
    random_state=42
)

rf_model.fit(X_train_rf, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test_rf)

# Evaluate
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print(f"R² Score: {rf_r2:.4f}")
print(f"Mean Absolute Error: {rf_mae:.4f}")
print(f"Root Mean Squared Error: {rf_rmse:.4f}")

# Feature importance
feature_names = ['DMSO Concentration', 'Cooling Rate']
importances = rf_model.feature_importances_
print("\nFeature Importance:")
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# Find optimal formulation using Random Forest
optimal_params_rf = []
best_viability_rf = 0

for dmso in np.linspace(0, 0.15, 30):
    for cooling in np.linspace(0.5, 10, 30):
        X_pred = imputer_rf.transform([[dmso, cooling]])
        pred_viability = rf_model.predict(X_pred)[0]
        if pred_viability > best_viability_rf:
            best_viability_rf = pred_viability
            optimal_params_rf = [dmso, cooling]

print(f"\nOptimal formulation (Random Forest):")
print(f"  DMSO: {optimal_params_rf[0]*100:.2f}%")
print(f"  Cooling Rate: {optimal_params_rf[1]:.2f} °C/min")
print(f"  Predicted Viability: {best_viability_rf*100:.2f}%")


# ============================================================
# MODEL 2: HISTOGRAM-BASED GRADIENT BOOSTING
# ============================================================
print("\n" + "="*60)
print("HISTOGRAM-BASED GRADIENT BOOSTING REGRESSOR")
print("="*60)

# HistGradientBoostingRegressor handles NaN natively (no imputation needed)
hgb_model = HistGradientBoostingRegressor(
    max_iter=100,          # Number of boosting iterations
    learning_rate=0.1,     # Shrinks contribution of each tree
    max_depth=5,           # Maximum depth of trees
    random_state=42
)

hgb_model.fit(X_train, y_train)

# Make predictions
hgb_pred = hgb_model.predict(X_test)

# Evaluate
hgb_r2 = r2_score(y_test, hgb_pred)
hgb_mae = mean_absolute_error(y_test, hgb_pred)
hgb_rmse = np.sqrt(mean_squared_error(y_test, hgb_pred))

print(f"R² Score: {hgb_r2:.4f}")
print(f"Mean Absolute Error: {hgb_mae:.4f}")
print(f"Root Mean Squared Error: {hgb_rmse:.4f}")

# Find optimal formulation using Histogram-based Gradient Boosting
optimal_params_hgb = []
best_viability_hgb = 0

for dmso in np.linspace(0, 0.15, 30):
    for cooling in np.linspace(0.5, 10, 30):
        pred_viability = hgb_model.predict([[dmso, cooling]])[0]
        if pred_viability > best_viability_hgb:
            best_viability_hgb = pred_viability
            optimal_params_hgb = [dmso, cooling]

print(f"\nOptimal formulation (Histogram-based Gradient Boosting):")
print(f"  DMSO: {optimal_params_hgb[0]*100:.2f}%")
print(f"  Cooling Rate: {optimal_params_hgb[1]:.2f} °C/min")
print(f"  Predicted Viability: {best_viability_hgb*100:.2f}%")


# ============================================================
# MODEL 3: NEURAL NETWORK
# ============================================================
print("\n" + "="*60)
print("NEURAL NETWORK (TENSORFLOW/KERAS)")
print("="*60)

# Impute NaN values before scaling (NN cannot handle NaN)
imputer_nn = SimpleImputer(strategy='mean')
X_train_imputed = imputer_nn.fit_transform(X_train)
X_test_imputed = imputer_nn.transform(X_test)

# Scale features for neural network (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Build neural network
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output layer
])

# Compile model
nn_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

# Train model
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=0  # Set to 1 to see training progress
)

# Make predictions
nn_pred = nn_model.predict(X_test_scaled, verbose=0)

# Evaluate
nn_r2 = r2_score(y_test, nn_pred)
nn_mae = mean_absolute_error(y_test, nn_pred)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))

print(f"R² Score: {nn_r2:.4f}")
print(f"Mean Absolute Error: {nn_mae:.4f}")
print(f"Root Mean Squared Error: {nn_rmse:.4f}")

# Find optimal formulation using Neural Network
optimal_params_nn = []
best_viability_nn = 0

for dmso in np.linspace(0, 0.15, 30):
    for cooling in np.linspace(0.5, 10, 30):
        X_pred_raw = np.array([[dmso, cooling]])
        X_pred_imputed = imputer_nn.transform(X_pred_raw)
        X_pred_scaled = scaler.transform(X_pred_imputed)
        pred_viability = nn_model.predict(X_pred_scaled, verbose=0)[0][0]
        if pred_viability > best_viability_nn:
            best_viability_nn = pred_viability
            optimal_params_nn = [dmso, cooling]

print(f"\nOptimal formulation (Neural Network):")
print(f"  DMSO: {optimal_params_nn[0]*100:.2f}%")
print(f"  Cooling Rate: {optimal_params_nn[1]:.2f} °C/min")
print(f"  Predicted Viability: {best_viability_nn*100:.2f}%")


# ============================================================
# SUMMARY AND COMPARISON
# ============================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

summary_df = pd.DataFrame({
    'Model': ['Random Forest', 'Hist Gradient Boosting', 'Neural Network'],
    'R² Score': [rf_r2, hgb_r2, nn_r2],
    'MAE': [rf_mae, hgb_mae, nn_mae],
    'RMSE': [rf_rmse, hgb_rmse, nn_rmse],
    'Optimal DMSO (%)': [optimal_params_rf[0]*100, optimal_params_hgb[0]*100, optimal_params_nn[0]*100],
    'Optimal Cooling (°C/min)': [optimal_params_rf[1], optimal_params_hgb[1], optimal_params_nn[1]],
    'Predicted Viability (%)': [best_viability_rf*100, best_viability_hgb*100, best_viability_nn*100]
})

print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('ML_Models_Comparison.csv', index=False)
print("\nComparison saved to: ML_Models_Comparison.csv")
