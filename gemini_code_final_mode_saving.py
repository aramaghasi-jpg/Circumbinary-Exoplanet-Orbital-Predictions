import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata, RegularGridInterpolator
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

# 1. Load and prepare Exoplanet Data
exoplanet_data = pd.read_csv("PS_2026.05.03_11.27.03.csv")
exoplanet_data = exoplanet_data[["pl_name", "hostname", "discoverymethod", "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbsmax", "pl_orbsmaxerr1", "pl_orbsmaxerr2", "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2", "pl_bmasselim", "pl_orbeccen", "pl_orbeccenerr1", "pl_orbeccenerr2", "pl_orbeccenlim", "st_rad", "st_raderr1", "st_raderr2", "st_radlim", "st_mass", "st_masserr1", "st_masserr2", "st_masslim"]]

refined_data = exoplanet_data[["pl_orbsmax", "pl_bmasse", "pl_rade", "st_rad", "st_mass", "pl_orbper", "pl_orbeccen"]]
refined_data = refined_data.interpolate()
refined_data = refined_data.fillna(refined_data.mean())

y = refined_data[["pl_orbsmax"]]
x = refined_data[["st_mass", "pl_orbper"]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def winsorize_data(df, lower_percentile=5, upper_percentile=95):
    df_winsorized = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_bound = df[col].quantile(lower_percentile / 100)
            upper_bound = df[col].quantile(upper_percentile / 100)
            df_winsorized[col] = np.clip(df[col], lower_bound, upper_bound)
    return df_winsorized

# Apply winsorization and scaling
X_train_winsorized = winsorize_data(X_train)
X_test_winsorized = winsorize_data(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_winsorized)
X_test_scaled = scaler.transform(X_test_winsorized)

# Train the Model (Updated to Support Vector Regressor)
my_svr_model = SVR(C=10, kernel='rbf')
my_svr_model.fit(X_train_scaled, y_train.values.ravel())

predictions = my_svr_model.predict(X_test_scaled)
print("Semimajor axis error:", mean_absolute_error(y_test, predictions))
print("Semimajor axis score:", r2_score(y_test, predictions))

# 2. Prepare the Grid Interpolator
ip = 0
data = np.genfromtxt("a_crit_Incl[%i].txt" % ip, delimiter=',', comments='#')
X_grid = data[:,0] # mu
Y_grid = data[:,1] # e_bin
Z_grid = data[:,2] # a_c/a_bin

xi = np.concatenate(([0.001], np.arange(0.01, 1, 0.01), [0.999]))
yi = np.arange(0, 0.81, 0.01)
zi = griddata((X_grid, Y_grid), Z_grid, (xi[:,None], yi[None,:]), method='linear', fill_value=0)

f = RegularGridInterpolator((xi, yi), zi)

# 3. Save the models using joblib
print("Saving models...")
joblib.dump(my_svr_model, 'svr_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(f, 'interpolator.joblib')
print("Models saved successfully!")