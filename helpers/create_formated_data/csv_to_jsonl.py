import pandas as pd
import json

from helpers.variables.paths import ORIGINAL_DATA_PATH, DATA_PATH

# python -m helpers.create_formated_data.csv_to_jsonl

x_train = ORIGINAL_DATA_PATH / "X_train.csv"
y_train = ORIGINAL_DATA_PATH / "y_train.csv"
output_path = DATA_PATH / "metadata.jsonl"

df_data = pd.read_csv(x_train)
df_label = pd.read_csv(y_train)

if len(df_data) != len(df_label):
    raise ValueError(f"Les fichiers ont un nombre de lignes différent : "
                        f"{len(df_data)} vs {len(df_label)}")

chunksize = 10000
a = 0
with open(output_path, "w", encoding="utf-8") as f:
    for df_data_chunk, df_label_chunk in zip(
        pd.read_csv(x_train, chunksize=chunksize),
        pd.read_csv(y_train, chunksize=chunksize)
    ):
        for i in range(len(df_data_chunk)):
            a += 1
            if a % 1000 == 0:
                print(a)
            record = {
                "data": df_data_chunk.iloc[i].to_dict(),
                "label": df_label_chunk.iloc[i].to_dict()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")