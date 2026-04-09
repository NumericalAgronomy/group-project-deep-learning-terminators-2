#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from transformers import (
    SNVTransformer,
    SavitzkyGolayTransformer,
    DerivativeTransformer,
)
# -------------------------------
# 1. Chargement et aperçu du dataset
# -------------------------------
# Remplacer le chemin si nécessaire
# -------------------------------
# 2. Identification des colonnes spectrales et de la colonne d'espèce
# -------------------------------
# On suppose que la colonne d'espèce s'appelle 'species'. 
# Si ce n'est pas le cas, modifiez la variable ci-dessous.
data_path = 'data/combined_data.csv'
df = pd.read_csv(data_path)

# -------------------------------
# 2. Identification des colonnes spectrales et de la colonne d'espèce
# -------------------------------
# On suppose que la colonne d'espèce s'appelle 'species'. 
# Si ce n'est pas le cas, modifiez la variable ci-dessous.
# Les colonnes spectrales sont toutes les colonnes sauf celle de l'espèce

species_col = 'class'
if species_col not in df.columns:
    print(f"La colonne '{species_col}' n'a pas été trouvée. " \
          "Veuillez modifier le nom de la colonne correspondant aux espèces.")
    species_col = df.columns[-1]  # On suppose ici que c'est la dernière colonne

spectral_columns = [col for col in df.columns if col != species_col]
X_spectral = df[spectral_columns].values
# Initialize transformers
snv_transformer = SNVTransformer()
sg_smoother = SavitzkyGolayTransformer(window_length=11, polyorder=3)
derivative1 = DerivativeTransformer(order=1, window_length=15, polyorder=3)
derivative2 = DerivativeTransformer(order=2, window_length=21, polyorder=3)
def fun(var):
    name = [k for k, v in globals().items() if v is var][0]
    return(name)

# Create transformed matrices
data_snv = pd.DataFrame(
    snv_transformer.fit_transform(X_spectral),
    columns=spectral_columns,
    index=df.index
)
data_smother = pd.DataFrame(
    sg_smoother.fit_transform(X_spectral),
    columns=spectral_columns,
    index=df.index
)
data_deriv1 = pd.DataFrame(
    derivative1.fit_transform(X_spectral),
    columns=spectral_columns,
    index=df.index
)
data_deriv2 = pd.DataFrame(
    derivative2.fit_transform(X_spectral),
    columns=spectral_columns,
    index=df.index
)
data_snv_transf = pd.DataFrame(
    sg_smoother.fit_transform(snv_transformer.fit_transform(X_spectral)),
    columns=spectral_columns,
    index=df.index
)
data_snv_deriv1 = pd.DataFrame(
    derivative1.fit_transform(snv_transformer.fit_transform(X_spectral)),
    columns=spectral_columns,
    index=df.index
)
data_snv_deriv2 = pd.DataFrame(
    derivative2.fit_transform(snv_transformer.fit_transform(X_spectral)),
    columns=spectral_columns,
    index=df.index
)
list_mats = [
    data_snv, data_smother, data_deriv1, 
    data_deriv2, data_snv_transf, data_snv_deriv1,data_snv_deriv2
]

for mat in list_mats:
    mat['class'] = df['class']


#ICI ON DEFINIT LE JEU DE DONNEES QU'ON VEUT

df=pd.read_csv(data_path)


species_col = 'class'
if species_col not in df.columns:
    print(f"La colonne '{species_col}' n'a pas été trouvée. " \
          "Veuillez modifier le nom de la colonne correspondant aux espèces.")
    species_col = df.columns[-1]  # On suppose ici que c'est la dernière colonne

spectral_columns = [col for col in df.columns if col != species_col]
X_spectral = df[spectral_columns].values

# -------------------------------
# 3. Analyse exploratoire
# -------------------------------

# 3.1 Distribution des espèces
species_counts = df[species_col].value_counts()
plt.figure(figsize=(8,6))
plt.bar(species_counts.index, species_counts.values, color='green')
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Distribution des espèces")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.2 Tracé de quelques courbes spectrales individuelles
plt.figure(figsize=(10,6))
# Sélection aléatoire de 10 échantillons (ou moins si le dataset est petit)
sample_df = df.sample(n=min(10, len(df)), random_state=42)
for idx, row in sample_df.iterrows():
    # Conversion des valeurs spectrales en flottant
    spectrum = row[spectral_columns].values.astype(float)
    plt.plot(spectral_columns, spectrum, label=row[species_col])
plt.xlabel("Longueur d'onde")
plt.ylabel("Réflectance / Intensité")
plt.title("Exemples de courbes spectrales")
plt.legend()
# Affichage des x-ticks avec un pas adapté
step = 10 if len(spectral_columns) > 10 else 1
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
plt.tight_layout()
plt.show()

# 3.3 Courbes moyennes par espèce avec intervalle de confiance (moyenne ± écart-type)
unique_species = df[species_col].unique()
plt.figure(figsize=(10,6))
for species in unique_species:
    subset = df[df[species_col] == species]
    spectra = subset[spectral_columns].astype(float)
    mean_spectrum = spectra.mean()
    std_spectrum = spectra.std()
    wavelengths = spectral_columns  # On suppose que le nom des colonnes représente la longueur d'onde
    plt.plot(wavelengths, mean_spectrum, label=f"{species} (moyenne)")
    plt.fill_between(wavelengths, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.2)
plt.xlabel("Longueur d'onde")
plt.ylabel("Réflectance / Intensité")
plt.title("Courbes spectrales moyennes par espèce")
plt.legend()
# Affichage des x-ticks avec un pas adapté
plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Analyse en Composantes Principales (PCA)
# -------------------------------
for mat in list_mats:
    # 3.4 Calcul et tracé de la première dérivée des courbes moyennes par espèce
    plt.figure(figsize=(10,6))
    # meow
    window_length = 7  # doit être impair
    polyorder = 2
    for species in unique_species:
        subset = mat[mat[species_col] == species]
        spectra = subset[spectral_columns].astype(float)
        mean_spectrum = spectra.mean().values
        # Vérification que window_length est adapté
        if window_length > len(mean_spectrum):
            window_length = len(mean_spectrum) if len(mean_spectrum) % 2 != 0 else len(mean_spectrum)-1
        plot = mean_spectrum
        plt.plot(spectral_columns, plot, label=f"{species} (transformée)")
    plt.xlabel("Longueur d'onde")
    plt.ylabel("réflectance transformée")
    plt.title(f"courbes spectrales moyennes par espèce pour  {fun(mat)}")
    plt.legend()
    # Affichage des x-ticks avec un pas adapté
    plt.xticks(ticks=np.arange(0, len(spectral_columns), step), labels=np.array(spectral_columns)[::step], rotation=45)
    plt.tight_layout()
    plt.show()


    # Application de la PCA sur les données spectrales
    spectra_data = mat[spectral_columns].astype(float)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(spectra_data)
    mat['PC1'] = pca_result[:, 0]
    mat['PC2'] = pca_result[:, 1]
    # Nouveaux logs pour afficher les résultats de la PCA
    print("\nRésultats de la PCA :")
    print(fun(mat))
    print("Rapport de variance expliquée : ", pca.explained_variance_ratio_)
    print("Aperçu de la projection PCA (5 premières lignes) :\n", mat[['PC1','PC2']].head())
    plt.figure(figsize=(8,6))
    for species in unique_species:
        subset = mat[mat[species_col] == species]
        plt.scatter(subset['PC1'], subset['PC2'], label=species)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Projection PCA des données spectrales pour {fun(mat)}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
   