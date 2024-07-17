import csv
import pandas as pd

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

district_mapping = {
    'Kavre palanchowk': 'Kavrepalanchowk',
    'Dhanusa': 'Dhanusha'
}

kpi1['District'] = kpi1['District'].replace(district_mapping)

combined_df = pd.merge(kpi1, kpi2, on  = 'District', how = 'outer')

#merged CSV file
combined_df.to_csv('combined_kpi.csv')
print(combined_df)
