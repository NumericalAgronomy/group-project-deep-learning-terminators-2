
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
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np


data_path = '/home/edelweiss/Documents/DATA_MANAGER/UE3/intro-to-random-forest-AnEdelweiss-3/data/combined_data.csv'
#df = pd.read_csv(data_path)
df = data_snv_deriv2

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
y = n_classes

validation_frames = []
for specie in df[species_col].unique():
    specie_df = df[df[species_col] == specie]
    if len(specie_df) >= 6:
        val_specie = specie_df.sample(n=8, random_state=23)
    else:
        val_specie = specie_df.sample(n=8, replace=True, random_state=15)
    validation_frames.append(val_specie)
validation_data = pd.concat(validation_frames)

# Jeu d'entraînement = données restantes
training_data = df.drop(validation_data.index)
#test
# Séparation en features et cible pour entraînement et validation
X_train = training_data.drop(columns=[species_col])
y_train = training_data[species_col]
X_test = validation_data.drop(columns=[species_col])
y_test = validation_data[species_col]
y_train_num_classe = y_train
y_test_num_classe = y_test

print(len(X_train),len(y_train),len(X_test),len(y_test))
## fait un random forest mais basé sur l'accuracy je crois 
cv_folds = 4
param_grid = {
    "n_estimators": [500],
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf": [1, 2],
}

model = RandomForestClassifier()

grid_search = GridSearchCV(
    model, param_grid, cv=cv_folds, scoring="accuracy", n_jobs=-1
)

grid_search.fit(X_train, y_train_num_classe)

cv_results = grid_search.cv_results_
rf_classification_best_score = grid_search.best_score_
rf_classification_best_param = grid_search.best_params_

print(f"Best score: {rf_classification_best_score} for {rf_classification_best_param}")

# On plot la matrice de confusion

model = RandomForestClassifier(**rf_classification_best_param)
model.fit(X_train, y_train_num_classe)


# MATRICE DE DISSIMILARITÉ ET MDS

# Extraction de la position des échantillons dans les feuilles de chaque arbre
feuilles = model.apply(X_train)

# Calcul de la matrice de dissimilarité (distance de Hamming)
dissimilarity_matrix = pairwise_distances(feuilles, metric='hamming')

# Application du Mds
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coordonnees_2d = mds.fit_transform(dissimilarity_matrix)

# graph
plt.figure(figsize=(10, 8))
scatter = plt.scatter(coordonnees_2d[:, 0], coordonnees_2d[:, 1], c=y_train_num_classe, cmap='viridis', s=50, alpha=0.8)
plt.title("Projection MDS basée sur la dissimilarité de la Random Forest")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(scatter, label='Classes de plantes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Use the model to predict the test set
rf_classification_y_pred = model.predict(X_test)
# Print the accuracy of the model
print("Accuracy on test: ", accuracy_score(y_test_num_classe, rf_classification_y_pred))

print("Balanced Accuracy on test: ", balanced_accuracy_score(y_test_num_classe, rf_classification_y_pred))
# Display confusion matrix and classification report
cm = confusion_matrix(y_test_num_classe, rf_classification_y_pred)

from sklearn.metrics import ConfusionMatrixDisplay
labels = np.unique(np.concatenate([y_test_num_classe.values, rf_classification_y_pred]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Greens', ax=ax, values_format='d')
ax.set_title("Confusion matrix - RandomForestClassifier")
plt.show()

print(classification_report(y_test_num_classe, rf_classification_y_pred))
print("Classification report:\n", cm)

# et les features
plt.figure(figsize=(8, 5))
plt.title("Feature's importance")
plt.bar(range(X_test.shape[1]), model.feature_importances_, align="center")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()