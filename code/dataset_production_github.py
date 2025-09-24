import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

BASE_PATH = './data/'

OUTPUT_PATH = join(BASE_PATH, './data/complete/')

# -----------------------------
# Load and preprocess dental data
# -----------------------------
df_den = pd.read_sas(join(BASE_PATH, '/examination/OHXDEN_J.XPT'))

for col in df_den.columns:
    if df_den[col].dtype == 'object' or df_den[col].dtype.name.startswith('bytes'):
        df_den[col] = df_den[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

df_den = df_den.replace('', np.nan)
df_den = df_den.astype({"SEQN": int}).set_index('SEQN')

df_den = df_den[df_den['OHDEXSTS'] == 1]

df_surf = pd.DataFrame(index=df_den.index)

for col in df_den.filter(regex='CSC').columns:
    df_den[col] = df_den[col].fillna('no caries').str.replace('-', '')
    df_surf[f'length_{col}'] = df_den[col].apply(lambda x: len(x) if x != 'no caries' else 0)
    df_den[col] = np.where(df_den[col].str.contains(r'\d', na=False), 'caries', df_den[col])

df_den = df_den.replace({'caries': 1.0, 'no caries': 0.0})
df_den_red = df_den.filter(regex='CSC').astype(float)
df_den_red['sum'] = df_den_red.sum(axis=1)

df_surf['surface_sum'] = df_surf.filter(regex='length_').sum(axis=1)
df_surf['tooth_count'] = df_den_red['sum']
df_surf['norm_sum'] = np.where(df_surf['tooth_count'] > 0,
                               (df_surf['surface_sum'] / df_surf['tooth_count']).round(2),
                               0.0)
df_surf['tooth_count_sev'] = df_surf['tooth_count']
df_surf['tooth_count'] = np.where(df_surf['tooth_count'] >= 1.0, 1.0, 0.0)

df_surf.to_csv(join(OUTPUT_PATH, 'pheno_2018_recent_upd_grok.csv'))

# -----------------------------
# Load and merge demographic data
# -----------------------------
df_demo = pd.read_sas(join(BASE_PATH, '/demo/DEMO_J.XPT'))
df_demo = df_demo.set_index('SEQN')
df_demo['outcome'] = df_surf['tooth_count']

list_to_keep = [
    'RIAGENDR','RIDAGEYR','RIDRETH1','RIDRETH3','DMDCITZN','DMDEDUC2',
    'DMDMARTL','SIALANG','DMDHHSIZ','DMDFMSIZ','DMDHHSZA','DMDHHSZB',
    'DMDHHSZE','DMDHRGND','WTINT2YR','DMDHREDZ','DMDHRAGZ','DMDHRMAZ',
    'WTMEC2YR','INDHHIN2','INDFMIN2','INDFMPIR','outcome'
]
df_demo = df_demo[list_to_keep]

# -----------------------------
# Merge function for SAS files
# -----------------------------
def merge_sas_files(path, df_main):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        df = pd.read_sas(join(path, f)).set_index('SEQN')
        for col in df.columns:
            if df[col].dtype not in [np.float64, np.int64]:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        for col in df.columns:
            df_main[col] = df[col]
    return df_main

df_demo = merge_sas_files(join(BASE_PATH, '/lab/'), df_demo)
df_demo = merge_sas_files(join(BASE_PATH, '/quest/'), df_demo)
df_demo = merge_sas_files(join(BASE_PATH, '/examination/'), df_demo)

nutrition_path = join(BASE_PATH, '/nutrition/')
onlyfiles = [f for f in listdir(nutrition_path) if isfile(join(nutrition_path, f)) and f.lower().endswith('.xpt')]
for f in onlyfiles:
    df = pd.read_sas(join(nutrition_path, f)).drop_duplicates(subset='SEQN').set_index('SEQN')
    for col in df.columns:
        if df[col].dtype not in [np.float64, np.int64]:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    for col in df.columns:
        df_demo[col] = df[col]

# -----------------------------
# Clean numeric data and remove invalid codes
# -----------------------------
df_demo = df_demo.select_dtypes(include=[np.float64, np.int64])

invalid_codes_small = [9, 7]
invalid_codes_medium = [77, 99, 777, 999]
invalid_codes_large = [7777, 9999, 77777, 99999, 5555]

for col in df_demo.columns:
    vals = df_demo[col].value_counts().index.tolist()
    if len(vals) < 9:
        for code in invalid_codes_small + invalid_codes_medium + invalid_codes_large:
            if code in vals:
                df_demo[col].replace(code, np.nan, inplace=True)
    elif len(vals) < 21:
        for code in invalid_codes_medium + invalid_codes_large:
            if code in vals:
                df_demo[col].replace(code, np.nan, inplace=True)
    else:
        for code in invalid_codes_large:
            if code in vals:
                df_demo[col].replace(code, np.nan, inplace=True)

with open(join(BASE_PATH, '/vars_to_remove.txt'), "r") as my_file:
    stripped_line = [s.rstrip() for s in my_file.readlines()]
df_demo.drop(stripped_line, axis=1, inplace=True)

df_demo.drop(columns=[col for col in df_demo.columns if df_demo[col].isnull().sum() > len(df_demo)*0.5], inplace=True)
df_demo.drop(columns=[col for col in df_demo.columns if 'OHX' in col], inplace=True)

# -----------------------------
# Handle categorical variables
# -----------------------------
categ_vars = [
    'RIDRETH1','RIDRETH3','DMDMARTL','DMDHRMAZ','DBQ229','HOQ065','DIQ010',
    'WHQ030','WHQ040','DR2DAY','DR2TWSZ','HUQ041','OHQ033'
]

for col in categ_vars:
    if col in df_demo.columns:
        dummies = pd.get_dummies(df_demo[col], prefix=col, dtype=float)
        df_demo = pd.concat([df_demo.drop(columns=[col]), dummies], axis=1)

# -----------------------------
# Save final dataset
# -----------------------------
df_demo.to_csv(join(OUTPUT_PATH, 'demo_lab_quest_nut_binary_2018.csv'))

