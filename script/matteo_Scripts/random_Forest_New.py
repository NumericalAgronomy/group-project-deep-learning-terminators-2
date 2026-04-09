# --- 1. Importations ---
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from dataLoading import data_snv, data_smother, data_deriv1, data_deriv2, data_snv_transf, data_snv_deriv1,data_snv_deriv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, r2_score,classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import statistics
# --- 2. Données simulées (remplacez par vos propres vecteurs spectraux) ---
# 300 échantillons x 20 bandes spectrales
data_path = 'data/combined_data.csv'
df = pd.read_csv(data_path)
#df = data_deriv2

species_col = 'class'
if species_col not in df.columns:
    print(f"La colonne '{species_col}' n'a pas été trouvée. " \
          "Veuillez modifier le nom de la colonne correspondant aux espèces.")
    species_col = df.columns[-1]  # On suppose ici que c'est la dernière colonne
target_col = df.columns[-1]
spectral_columns = [col for col in df.columns if col != species_col]
spectra = df[spectral_columns].values
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])
n_classes = df[target_col].nunique()
# Concatenate all transformed features

X = spectra
y = df["class"]

# Répartition équilibrée (optionnel)
#from collections import Counter
#n_samples_per_class = 6
#X = np.vstack([X] * n_samples_per_class).reshape(-1, 20)
#y = np.array([i for i in np.arange(n_samples_per_class)] for _ in range(6))
#y = np.array([i for i in range(6)] * n_samples_per_class)

# --- 3. Préparation des données ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Normalisation (optionnel mais recommandé)
window = 17
poly = 2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = savgol_filter(X_train_scaled, window_length=window, polyorder=poly, deriv=2, axis=1)
X_test_scaled = savgol_filter(X_test_scaled, window_length=window, polyorder=poly, deriv=2, axis=1)

# --- 4. Entraînement du modèle Random Forest ---
clf = RandomForestClassifier(n_estimators=500, random_state=42,n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# --- 5. Évaluation ---
y_pred = clf.predict(X_test_scaled)

print("Précision :", accuracy_score(y_test, y_pred))
print("Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['A', 'B', 'C', 'D', 'E', 'F',"G"]))

# --- 6. Utilisation pour une nouvelle prédiction ---
# nouvelle_echantillon = np.array([[1, 2, 3, ...]])  # votre vecteur spectral
# prediction = clf.predict([nouvelle_echantillon_scaled])
