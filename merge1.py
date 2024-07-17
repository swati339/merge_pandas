import pandas as pd
import difflib

data_kpi1 = {
    'District': ['Kathmandu', 'Kavre palanchowk', 'Dhanusa'],
    'KPI_1': [0.8, 0.75, 0.85]
}

kpi1 = pd.DataFrame(data_kpi1)

data_kpi2 = {
   'District': ['Kathmandu', 'Kavrepalanchowk', 'Dhanusha'],
    'KPI_2': [0.35, 0.65, 0.6]
}

kpi2 = pd.DataFrame(data_kpi2)

# Function to find the best match using difflib
def find_match(name, options):
    matches = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Map district names in kpi1 to those in kpi2 using difflib
kpi1['Mapped_District'] = kpi1['District'].apply(lambda x: find_match(x, kpi2['District']))

# Merge based on mapped district names
combined_df = pd.merge(kpi1, kpi2, left_on='Mapped_District', right_on='District', how='outer')

combined_df.drop(['Mapped_District', 'District_y'], axis=1, inplace=True)
combined_df.rename(columns={'District_x': 'District'}, inplace=True)

# Fill missing values with NaN
combined_df.fillna('', inplace=True)

#merged CSV
combined_df.to_csv('combined_kpi.csv')
print(combined_df[['District', 'KPI_1', 'KPI_2']].to_string(index=False))

