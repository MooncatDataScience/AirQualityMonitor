import os
import pandas as pd

root_folder = "F://Ds//air-pollution-api//develop//"
folder_names = [
    "中部空品區_2022", 
    "北部空品區_2022",
    "竹苗空品區_2022",
    "宜蘭空品區_2022",
    "花東空品區_2022",
    "高屏空品區_2022",
    "雲嘉南空品區_2022",
    "離島_2022"
]

items = []

for folder_name in folder_names:
    folder_path = os.path.join(root_folder, folder_name)
    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        for csv_file in os.listdir(file_path):
            csv_path = os.path.join(file_path, csv_file)
            items.append(csv_path)

dp = [pd.read_csv(item) for item in items[:-1]]
merged_data = pd.concat(dp)

merged_data.to_csv('merged.csv', index=False)
