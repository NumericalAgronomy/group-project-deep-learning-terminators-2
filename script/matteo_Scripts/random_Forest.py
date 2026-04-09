
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import statistics
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.metrics import ConfusionMatrixDisplay
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
y = n_classes

validation_frames = []
for specie in df[species_col].unique():
    specie_df = df[df[species_col] == specie]
    if len(specie_df) >= 6:
        val_specie = specie_df.sample(n=7, random_state=36)
    else:
        val_specie = specie_df.sample(n=5, replace=True, random_state=15)
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

window = 17
poly = 2

X_train = savgol_filter(X_train, window_length=window, polyorder=poly, deriv=2, axis=1)
X_test = savgol_filter(X_test, window_length=window, polyorder=poly, deriv=2, axis=1)

y_train_num_classe = y_train
y_test_num_classe = y_test

print(len(X_train),len(y_train),len(X_test),len(y_test))
## fait un random forest mais basé sur l'accuracy je crois 
cv_folds = 4
param_grid = {
    "n_estimators": [500],
    "max_depth": [None],
    "min_samples_split": [5],
    "min_samples_leaf": [1],
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
outputForet = [0]*50
for i in range(0, 50):
    model = RandomForestClassifier(**rf_classification_best_param)
    model.fit(X_train, y_train_num_classe)
    # Use the model to predict the test set
    rf_classification_y_pred = model.predict(X_test)
    # Print the accuracy of the model
    print("Accuracy on test: ", accuracy_score(y_test_num_classe, rf_classification_y_pred))
    print("Balanced Accuracy on test: ", balanced_accuracy_score(y_test_num_classe, rf_classification_y_pred))
    #outputForet.append(accuracy_score(y_test_num_classe, rf_classification_y_pred))
    outputForet[i]=(accuracy_score(y_test_num_classe, rf_classification_y_pred))
    print(statistics.mean(outputForet))
    print(i)

print(statistics.mean(outputForet))
# Display confusion matrix and classification report
cm = confusion_matrix(y_test_num_classe, rf_classification_y_pred)

labels = np.unique(np.concatenate([y_test_num_classe.values, rf_classification_y_pred]))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Greens', ax=ax, values_format='d')
ax.set_title("Confusion matrix - RandomForestClassifier")
plt.show()

print(classification_report(y_test_num_classe, rf_classification_y_pred))
print("Classification report:\n", cm)

# # et les features
# plt.figure(figsize=(8, 5))
# plt.title("Feature's importance")
# plt.bar(spectral_columns, model.feature_importances_, align="center")
# plt.xlabel("Features")
# plt.ylabel("Importance Score")
# plt.show()
#meilleur graphique
step_size = 10
selected_wavelengths = spectral_columns[::step_size]
selected_importances = model.feature_importances_[::step_size]
plt.figure(figsize=(14, 6)) 
plt.title(f"Feature's importance (Sampled every {step_size}th Wavelength)")
plt.bar(selected_wavelengths, selected_importances, align="center")
plt.xlabel("Wavelength (Features)")
plt.ylabel("Importance Score")

plt.grid(axis='y', alpha=0.5) 
plt.show()
# MATRICE DE DISSIMILARITÉ ET MDS

# Extraction de la position des échantillons dans les feuilles de chaque arbre
feuilles = model.apply(X_train)

# Calcul de la matrice de dissimilarité (distance de Hamming)
dissimilarity_matrix = pairwise_distances(feuilles, metric='hamming')

# Application du Mds
mds = MDS(n_components=2, metric='precomputed', random_state=42,init='classical_mds',n_init=1)
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
