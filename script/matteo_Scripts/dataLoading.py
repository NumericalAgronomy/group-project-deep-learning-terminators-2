#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from transformers import (
    SNVTransformer,
    SavitzkyGolayTransformer,
    DerivativeTransformer,
)
# -------------------------------
# 1. Chargement et aperçu du dataset
# -------------------------------
# Remplacer le chemin si nécessaire
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