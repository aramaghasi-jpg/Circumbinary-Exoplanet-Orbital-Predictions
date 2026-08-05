import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import time
import astropy.units as u
from scipy.interpolate import griddata, RegularGridInterpolator
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

# Download the data:
exoplanet_data = pd.read_csv("PS_2026.08.05_14.00.09.csv")

print(exoplanet_data.keys())
exoplanet_data.describe()

# Ensure we create a copy to avoid SettingWithCopy warnings
refined_data = exoplanet_data[["pl_orbsmax", "pl_bmasse", "pl_rade", "st_rad", "st_mass", "pl_orbper", "pl_orbeccen"]].copy()

# ==========================================
# DATA CLEANING & CONVERSION LOGIC
# ==========================================

# 1. Fill missing stellar masses with the mean so Kepler's Third Law can be calculated
st_mass_filled = refined_data['st_mass'].fillna(refined_data['st_mass'].mean())

# 2. Identify rows where 'pl_orbsmax' is missing, but 'pl_orbper' is present
conversion_mask = refined_data['pl_orbsmax'].isna() & refined_data['pl_orbper'].notna()

# 3. Convert period (days) to years and calculate semimajor axis (AU)
period_years = refined_data.loc[conversion_mask, 'pl_orbper'] / 365.25
refined_data.loc[conversion_mask, 'pl_orbsmax'] = (period_years**2 * st_mass_filled.loc[conversion_mask])**(1./3.)

# 4. Drop any rows that STILL do not have a semimajor axis (meaning both were NaN)
refined_data = refined_data.dropna(subset=['pl_orbsmax'])

# ==========================================

# Interpolate and fill any remaining NaNs in other columns
refined_data = refined_data.interpolate()
refined_data = refined_data.fillna(refined_data.mean())

null_counts_orb = np.sum(refined_data["pl_orbsmax"].isnull())
null_counts_mass = np.sum(refined_data["pl_bmasse"].isnull())
null_counts_rad_e = np.sum(refined_data["pl_rade"].isnull())
null_counts_rad = np.sum(refined_data["st_rad"].isnull())
null_counts_st_mass = np.sum(refined_data["st_mass"].isnull())

print("Null counts for orbit:", null_counts_orb)
print("Null counts for mass:", null_counts_mass)
print("Null counts for planet radius:", null_counts_rad_e)
print("Null counts for star mass:", null_counts_rad)
print("Null counts for stellar mass:", null_counts_st_mass)

# Split the entire DataFrame first to keep track of test case planet mass and radius
train_data, test_data = train_test_split(refined_data, test_size=0.2, random_state=42)

# UPDATED FEATURE SELECTION
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

# Apply winsorization to X_train and X_test
X_train_winsorized = winsorize_data(X_train)
X_test_winsorized = winsorize_data(X_test)

print("X_train_winsorized dimensions:", X_train_winsorized.shape)
print("X_test_winsorized dimensions:", X_test_winsorized.shape)

my_random_forest_model = RandomForestRegressor(max_depth=11, n_estimators=20, random_state=0)
my_random_forest_model.fit(X_train_winsorized, y_train.values.ravel()) 
predictions = my_random_forest_model.predict(X_test_winsorized)

mySMA_test = y_test.values.ravel()
mySMA_predictions = predictions

mySMA_error = mean_absolute_error(mySMA_test, mySMA_predictions)
mySMA_score = r2_score(mySMA_test, mySMA_predictions)

print("Semimajor axis error:", mySMA_error)
print("Semimajor axis score:", mySMA_score)

# ==========================================
# ERROR PLOTTING LOGIC
# ==========================================
# Calculate absolute error for each test case
test_errors = np.abs(mySMA_test - mySMA_predictions)

# Retrieve mass and radius for the test cases
test_masses = test_data["pl_bmasse"]
test_radii = test_data["pl_rade"]

# Create the plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(test_masses, test_radii, c=test_errors, cmap='coolwarm', alpha=0.8, edgecolor='k')
plt.colorbar(scatter, label='Absolute Error in Predicted SMA (AU)')
plt.xscale('log') 
plt.yscale('log') 
plt.xlabel('Planet Mass (Earth Masses)')
plt.ylabel('Planet Radius (Earth Radii)')
plt.title('Model Prediction Error by Planet Mass and Radius')
plt.grid(True, which="both", ls="--", alpha=0.2)
plt.show()
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

M_tot = 0.972 + 1.133 # Total star mass in M_sun
mu = 0.972/(0.972 + 1.133)  
e_bin = 0.524
P_bin = 79.91

a_bin = (P_bin**2*M_tot)**(1./3)

print("a_c = %1.3f AU" % (get_stability_limit(f,mu,e_bin)*a_bin))

#User inputs
Star_1_mass = 2.15
Star_1_radius = 1.50 # ADDED: Required for the updated model features
Star_2_mass = 1.72
planet_mass = 125
planet_radius = 6.5
e_bin_user = 0.01
P_bin_user = 75

#Processed user inputs
M_tot_user = Star_1_mass + Star_2_mass
mu_user = Star_1_mass/(Star_1_mass + Star_2_mass)
a_bin_user = (P_bin_user**2*M_tot_user)**(1./3)

#SMA Evaluation Model
# Create a DataFrame from user input, matching the structure of X_train
user_input_df = pd.DataFrame([[planet_radius, planet_mass, Star_1_radius, Star_1_mass]], columns=X_train.columns)

# Winsorize the user input
user_input_winsorized = winsorize_data(user_input_df)

# Make prediction
a_predicted = my_random_forest_model.predict(user_input_winsorized)[0]
print(f"\nPredicted Semimajor Axis: {a_predicted:.4f} AU")

#CSP Stability Model
a_c = get_stability_limit(f,mu_user,e_bin_user)*a_bin_user
print("a_c = %1.3f AU" % (a_c))
pl_valid = a_c > a_predicted

if pl_valid:
  print("This orbit is stable!")
else:
  print("This orbit is unstable!")