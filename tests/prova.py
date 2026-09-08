from pathlib import Path
from tkinter.font import names
import numpy as np

import pandas as pd
import re
from datetime import datetime

TAULA_SUPLEMENTARIA_PATH = Path('/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/41597_2023_1928_MOESM1_ESM.xlsx')

df_raw = pd.read_excel(TAULA_SUPLEMENTARIA_PATH)
df = pd.read_excel(TAULA_SUPLEMENTARIA_PATH,header=3)
df.columns= ['name', 'study_date', 'series_uid',
              'series_description', 'phase', 'n_images']
df['name'] = df['name'].astype(str).str.strip()

for pid in ['HCC_016','HCC_018','HCC_031','HCC_037','HCC_056','HCC_072','HCC_082']:
    print(f"\n=== {pid} ===")
    print(df[df['name']==pid][['study_date','series_description','phase']].to_string())

print(df[df['name']=='HCC_042'][['series_description','phase','n_images']].to_string())

#def is_uid(text):
#    if not isinstance(text, str):
#        return False
#    return bool(re.match(r'^[\d.]+$', text)) and '.' in text
#
#df['series_uid'] = df['series_uid'].astype(object)
## Per cada fila, mira si la 'description' es un UID
#mask = df['series_description'].apply(is_uid)
#
## On hi havia un UID a description, mou-lo a series_uid
#df.loc[mask, 'series_uid'] = df.loc[mask, 'series_description']
#
## I deixa description com a NaN allà
#df.loc[mask, 'series_description'] = np.nan
##print(df.shape)
##print(df.columns.tolist())
##print(df.head())
##print(df['name'].nunique())
##print(df['phase'].value_counts())
#
##filtrem per fases separades
#fases_separades = ['pre-contrast', 'arterial','PV']
#df = df[df['phase'].isin(fases_separades)]
#print(df.head())
#
##filtrem per data
#df['study_date'] = pd.to_datetime(df['study_date'])
#df_pre_tt = df[df.groupby('name')['study_date'].transform('min')==df['study_date']]
#
#
#
#pacients = {}
#for name, group in df_pre_tt.groupby('name'):
#    pacients[name] = {
#        'name': name,
#        'serie': group.drop(columns=['name']).to_dict('records')}
#
#pacients = {
#    pid: dades
#    for pid, dades in pacients.items()
#    if len(dades['serie']) >= 2
#}
#
#print(pacients)
#
#
#