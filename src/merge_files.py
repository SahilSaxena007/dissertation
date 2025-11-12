import pandas as pd

# Step 1: Read the ADNIMERGE file and select the specified columns
adnimerge = pd.read_csv('../Study_files/ADNIMERGE_14Oct2024.csv')
selected_columns = ['RID', 'VISCODE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'TAU', 'PTAU', 'MMSE', 'DX', 'ABETA']
basefile = adnimerge[selected_columns]


# Change 'CN' to 'SCD' and 'Dementia' to 'AD' in the 'DX' column
basefile['DX'] = basefile['DX'].replace({'Dementia': 'AD', 'CN': 'SCD'}) 

# Step 3: Read the BLENNOWPLASMANFL file and merge with the basefile
blennowplasmatau = pd.read_csv('../Study_files/ADNI_BLENNOWPLASMANFL_10_03_18_11Oct2024.csv')
merged_data = pd.merge(basefile, blennowplasmatau[['RID', 'VISCODE', 'PLASMA_NFL']], 
                       on=['RID', 'VISCODE'], how='left')

# Step 4: Read the UPENNMSMSABETA2 file, calculate AB4240 ratio, and merge with the previously merged data
upennmsmsabeta2 = pd.read_csv('../Study_files/UPENNPLASMA_14Oct2024.csv')
upennmsmsabeta2['AB4240'] = upennmsmsabeta2['AB42'] / upennmsmsabeta2['AB40']
merged_data = pd.merge(merged_data, upennmsmsabeta2[['RID', 'VISCODE', 'AB4240']], 
                       on=['RID', 'VISCODE'], how='left')

# Step 5: Read the BLENNOWPLASMATAU file and merge with the previously merged data
blennowplasmatau = pd.read_csv('../Study_files/BLENNOWPLASMATAU_14Oct2024.csv')
merged_data = pd.merge(merged_data, blennowplasmatau[['RID', 'VISCODE', 'PLASMATAU']], 
                       on=['RID', 'VISCODE'], how='left')

merged_data = merged_data.dropna(subset=['PLASMA_NFL'], how='any')

# # Save the merged data
merged_data.to_csv('../data/merged_data.csv', index=False)
print("\nMerged data saved to '../data/merged_data.csv'")
