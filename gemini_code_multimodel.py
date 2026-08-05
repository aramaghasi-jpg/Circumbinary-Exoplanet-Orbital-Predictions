import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import re
import time
import astropy.units as u
from scipy.interpolate import griddata, RegularGridInterpolator
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

# Download the data:
exoplanet_data = pd.read_csv("PS_2026.08.05_14.00.09.csv")

# Ensure we create a copy to avoid SettingWithCopy warnings
refined_data = exoplanet_data[["pl_orbsmax", "pl_bmasse", "pl_rade", "st_rad", "st_mass", "pl_orbper", "pl_orbeccen"]].copy()

# ==========================================
# DATA CLEANING & CONVERSION LOGIC
# ==========================================
st_mass_filled = refined_data['st_mass'].fillna(refined_data['st_mass'].mean())
conversion_mask = refined_data['pl_orbsmax'].isna() & refined_data['pl_orbper'].notna()
period_years = refined_data.loc[conversion_mask, 'pl_orbper'] / 365.25
refined_data.loc[conversion_mask, 'pl_orbsmax'] = (period_years**2 * st_mass_filled.loc[conversion_mask])**(1./3.)
refined_data = refined_data.dropna(subset=['pl_orbsmax'])

refined_data = refined_data.interpolate()
refined_data = refined_data.fillna(refined_data.mean())

# Split the entire DataFrame first
train_data, test_data = train_test_split(refined_data, test_size=0.2, random_state=42)

X_train = train_data[["pl_rade", "pl_bmasse", "st_rad", "st_mass"]]
y_train = train_data[["pl_orbsmax"]]
X_test = test_data[["pl_rade", "pl_bmasse", "st_rad", "st_mass"]]
y_test = test_data[["pl_orbsmax"]]

def winsorize_data(df, lower_percentile=5, upper_percentile=95):
    df_winsorized = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_bound = df[col].quantile(lower_percentile / 100)
            upper_bound = df[col].quantile(upper_percentile / 100)
            df_winsorized[col] = np.clip(df[col], lower_bound, upper_bound)
    return df_winsorized

# Apply winsorization
X_train_winsorized = winsorize_data(X_train)
X_test_winsorized = winsorize_data(X_test)
y_train_flat = y_train.values.ravel()
y_test_flat = y_test.values.ravel()

# ==========================================
# HYPERPARAMETER TUNING & MODEL SELECTION
# ==========================================
# Define the models and the parameter grids to search
model_grids = {
    "Random Forest": {
        "estimator": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None]
        }
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    },
    "Support Vector Regressor": {
        # SVR requires scaling to function properly. We use a Pipeline to prevent data leakage.
        "estimator": Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ]),
        "params": {
            "svr__C": [0.1, 1, 10],
            "svr__kernel": ['rbf', 'linear']
        }
    }
}

best_models = {}
predictions_dict = {}

print("Starting Hyperparameter Tuning. This may take a moment...\n")
print("-" * 50)

# Iterate over models, run GridSearchCV, and evaluate
for model_name, config in model_grids.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["params"],
        cv=3,
        scoring='r2',
        n_jobs=-1 # Uses all available CPU cores for faster processing
    )
    
    grid_search.fit(X_train_winsorized, y_train_flat)
    best_model = grid_search.best_estimator_
    
    # Store the best model and make predictions
    best_models[model_name] = best_model
    preds = best_model.predict(X_test_winsorized)
    predictions_dict[model_name] = preds
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_flat, preds)
    r2 = r2_score(y_test_flat, preds)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("-" * 50)

# Select the absolute best model overall based on R2 score to use for the Stability Evaluation later
best_overall_name = max(predictions_dict, key=lambda k: r2_score(y_test_flat, predictions_dict[k]))
best_overall_model = best_models[best_overall_name]
print(f"\nWINNER: {best_overall_name} chosen for final stability evaluation.\n")

# ==========================================
# ERROR PLOTTING LOGIC (SUBPLOTS)
# ==========================================
test_masses = test_data["pl_bmasse"]
test_radii = test_data["pl_rade"]

# Create 1 row, 3 columns of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Model Prediction Error Comparison by Planet Mass and Radius', fontsize=16)

for ax, (model_name, preds) in zip(axes, predictions_dict.items()):
    test_errors = np.abs(y_test_flat - preds)
    
    scatter = ax.scatter(test_masses, test_radii, c=test_errors, cmap='coolwarm', alpha=0.8, edgecolor='k')
    ax.set_xscale('log') 
    ax.set_yscale('log') 
    ax.set_xlabel('Planet Mass (Earth Masses)')
    ax.set_title(f'{model_name}\nR2: {r2_score(y_test_flat, preds):.4f}')
    ax.grid(True, which="both", ls="--", alpha=0.2)
    
axes[0].set_ylabel('Planet Radius (Earth Radii)')
fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Absolute Error in Predicted SMA (AU)')
plt.show()

# ==========================================
# BINARY STAR STABILITY EVALUATION
# ==========================================
ip = 0
data = np.genfromtxt("a_crit_Incl[%i].txt" % ip,delimiter=',',comments='#')

X_bin = data[:,0] #mu
Y_bin = data[:,1] #e_bin
Z_bin = data[:,2] #a_c/a_bin

xi = np.concatenate(([0.001],np.arange(0.01,1,0.01),[0.999]))
yi = np.arange(0,0.81,0.01)
zi = griddata((X_bin,Y_bin),Z_bin,(xi[:,None],yi[None,:]),method = 'linear',fill_value=0)

f = RegularGridInterpolator((xi, yi), zi) 

def get_stability_limit(f,mu,e_bin):
    return f([[mu, e_bin]])[0]

M_tot = 0.972 + 1.133 
mu = 0.972/(0.972 + 1.133)  
e_bin = 0.524
P_bin = 79.91

a_bin = (P_bin**2*M_tot)**(1./3)

print("a_c = %1.3f AU" % (get_stability_limit(f,mu,e_bin)*a_bin))

#User inputs
Star_1_mass = 2.15
Star_1_radius = 1.50 
Star_2_mass = 1.72
planet_mass = 125
planet_radius = 6.5
e_bin_user = 0.01
P_bin_user = 75

#Processed user inputs
M_tot_user = Star_1_mass + Star_2_mass
mu_user = Star_1_mass/(Star_1_mass + Star_2_mass)
a_bin_user = (P_bin_user**2*M_tot_user)**(1./3)

#SMA Evaluation Model - USING THE BEST TUNED MODEL
user_input_df = pd.DataFrame([[planet_radius, planet_mass, Star_1_radius, Star_1_mass]], columns=X_train.columns)
user_input_winsorized = winsorize_data(user_input_df)

a_predicted = best_overall_model.predict(user_input_winsorized)[0]
print(f"\nPredicted Semimajor Axis (using {best_overall_name}): {a_predicted:.4f} AU")

#CSP Stability Model
a_c = get_stability_limit(f,mu_user,e_bin_user)*a_bin_user
print("a_c = %1.3f AU" % (a_c))
pl_valid = a_c > a_predicted

if pl_valid:
  print("This orbit is stable!")
else:
  print("This orbit is unstable!")