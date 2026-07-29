import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import time
import astropy.units as u
import numpy as np  #We need numpy for the convenience function genfromtxt
from scipy.interpolate import griddata,RegularGridInterpolator  #We need this to map our data to a grid and interpolate between grid points
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

#Download the data:

exoplanet_data = pd.read_csv("PS_2026.05.03_11.27.03.csv")

exoplanet_data = exoplanet_data[["pl_name", "hostname", "discoverymethod", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmasselim", "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim", "st_rad", "st_raderr1", "st_raderr2", "st_radlim", "st_mass", "st_masserr1", "st_masserr2", "st_masslim"]]
print(exoplanet_data.keys())
exoplanet_data.describe()

refined_data = exoplanet_data[["pl_orbsmax", "pl_bmasse", "pl_rade", "st_rad", "st_mass", "pl_orbper", "pl_orbeccen"]]

refined_data = refined_data.interpolate()
# Fill any remaining NaNs (e.g., at the beginning/end of series) with the mean
# This ensures no NaNs are passed to subsequent steps
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

y = refined_data[["pl_orbsmax"]]
print(y)

x = refined_data[["st_mass", "pl_orbper"]]
print(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the winsorized training data and transform both winsorized training and test data
X_train_scaled = scaler.fit_transform(X_train_winsorized)
X_test_scaled = scaler.transform(X_test_winsorized)

print("X_train_scaled dimensions:", X_train_scaled.shape)
print("X_test_scaled dimensions:", X_test_scaled.shape)

print("X_train dimensions:", X_train.shape)
print("X_test dimensions:", X_test.shape)
print("y_train dimensions:", y_train.shape)
print("y_test dimensions:", y_test.shape)

from sklearn.ensemble import RandomForestRegressor
my_random_forest_model = RandomForestRegressor(max_depth=11, n_estimators=20, random_state=0)
my_random_forest_model.fit(X_train_scaled, y_train.values.ravel()) # .ravel() to convert y_train to 1D array
predictions = my_random_forest_model.predict(X_test_scaled)

mySMA_test = y_test
mySMA_predictions = predictions

mySMA_error = mean_absolute_error(mySMA_test, mySMA_predictions)
mySMA_score = r2_score(mySMA_test, mySMA_predictions)


print("Semimajoral axis error", mySMA_error)
print("Semimajoral axis score", mySMA_score)

ip = 0
data = np.genfromtxt("a_crit_Incl[%i].txt" % ip,delimiter=',',comments='#')  #The data contained in this repository

X = data[:,0] #mu
Y = data[:,1] #e_bin
Z = data[:,2] #a_c/a_bin

xi = np.concatenate(([0.001],np.arange(0.01,1,0.01),[0.999]))
yi = np.arange(0,0.81,0.01)
zi = griddata((X,Y),Z,(xi[:,None],yi[None,:]),method = 'linear',fill_value=0)  #make the grid

f = RegularGridInterpolator((xi, yi), zi) # make the 2d interpolation

def get_stability_limit(f,mu,e_bin):
    return f([[mu, e_bin]])[0]

M_tot = 0.972 + 1.133 # Total star mass in M_sun
mu = 0.972/(0.972 + 1.133)  #converting from M_A, M_B --> mu  Pourbaix & Boffin (2016)
e_bin = 0.524
P_bin = 79.91

a_bin = (P_bin**2*M_tot)**(1./3)

print("a_c = %1.3f AU" % (get_stability_limit(f,mu,e_bin)*a_bin))

#User inputs
Star_1_mass = 2.15
Star_2_mass = 1.72
planet_mass = 125
planet_radius = 6.5
planet_period = 50
e_bin = 0.01
P_bin = 75

#Processed user inputs
M_tot = Star_1_mass + Star_2_mass
mu = Star_1_mass/(Star_1_mass + Star_2_mass)
a_bin = (P_bin**2*M_tot)**(1./3)

#SMA Evaluation Model
# Create a DataFrame from user input, matching the structure of X_train
user_input_df = pd.DataFrame([[Star_1_mass, planet_period]], columns=x.columns)
# Winsorize the user input (using the same function as for training data)
user_input_winsorized = winsorize_data(user_input_df)
# Scale the user input (using the same scaler fitted on training data)
user_input_scaled = scaler.transform(user_input_winsorized)
# Make prediction
a_predicted = my_random_forest_model.predict(user_input_scaled)[0]
print(f"\nPredicted Semimajoral Axis: {a_predicted:.4f} AU")

#CSP Stability Model
a_c = get_stability_limit(f,mu,e_bin)*a_bin
print("a_c = %1.3f AU" % (a_c))
pl_valid = a_c > a_predicted
if pl_valid:
  print("This orbit is stable!")
else:
  print("This orbit is unstable!")

