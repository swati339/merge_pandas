import csv
import pandas as pd

data_kpi1 = [
    ["District", "KPI_1"],
    ["Kathmandu", 0.8],
    ["Kavre palanchowk", 0.75],
    ["Dhanusa", 0.85]
]

data_kpi2 = [
    ["District", "KPI_2"],
    ["Kathmandu", 0.35],
    ["Kavrepalanchowk", 0.65],
    ["Dhanusha", 0.6]
]

def write_csv(file_name, data):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

write_csv('kpi1.csv', data_kpi1)
write_csv('kpi2.csv', data_kpi2)

kpi1 = pd.read_csv('kpi1.csv')
kpi2 = pd.read_csv('kpi2.csv')

combined_df = pd.merge(kpi1, kpi2, left_on='District', right_on='District')
print(combined_df)
