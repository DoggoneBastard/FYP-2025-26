
# ============================================================
# STEP 2: DIFFERENTIAL EVOLUTION OPTIMIZATION
# ============================================================
# This script uses differential evolution to optimize cryopreservation formulations

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv('MSC_Cleaned_Data.csv')

# Prepare features and target
# For this example, we'll use DMSO and Cooling Rate as features
X = df[['DMSO_Numeric', 'Cooling_Rate_Numeric']].values
y = df['Viability_Numeric'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a surrogate model (Random Forest) on existing data
surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
surrogate_model.fit(X_train, y_train)

print(f"Surrogate model R² score: {surrogate_model.score(X_test, y_test):.3f}")

# Define objective function for differential evolution
def objective_function(params):
    """
    Objective function to maximize viability (minimize negative viability)
    params[0] = DMSO concentration (0 to 0.15 for DMSO-free focus)
    params[1] = Cooling rate (0.5 to 10 °C/min)
    """
    # Reshape params for prediction
    X_pred = np.array([params])

    # Predict viability using surrogate model
    predicted_viability = surrogate_model.predict(X_pred)[0]

    # We minimize negative viability (to maximize viability)
    return -predicted_viability

# Define bounds for optimization
# [DMSO concentration, Cooling rate]
bounds = [(0, 0.15),      # DMSO: 0% to 15%
          (0.5, 10)]      # Cooling rate: 0.5 to 10 °C/min

# Run differential evolution
result = differential_evolution(
    objective_function,
    bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=15,
    seed=42
)

# Display results
print("\n" + "="*60)
print("DIFFERENTIAL EVOLUTION RESULTS")
print("="*60)
print(f"Optimal DMSO concentration: {result.x[0]:.4f} ({result.x[0]*100:.2f}%)")
print(f"Optimal cooling rate: {result.x[1]:.2f} °C/min")
print(f"Predicted maximum viability: {-result.fun:.4f} ({-result.fun*100:.2f}%)")
print("="*60)

# Save results
results_df = pd.DataFrame({
    'Parameter': ['DMSO Concentration', 'Cooling Rate', 'Predicted Viability'],
    'Value': [f"{result.x[0]*100:.2f}%", f"{result.x[1]:.2f} °C/min", f"{-result.fun*100:.2f}%"]
})
results_df.to_csv('DE_Optimization_Results.csv', index=False)
print("\nResults saved to: DE_Optimization_Results.csv")