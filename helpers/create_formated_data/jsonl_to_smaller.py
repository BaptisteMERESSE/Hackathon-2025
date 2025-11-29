import json

from helpers.variables.paths import DATASET_PATH, SMALLER_DATASET_PATH

# python -m helpers.create_formated_data.jsonl_to_smaller


input_file = DATASET_PATH / "metadata.jsonl"

n_splits = 20  # nombre de fichiers de sortie

# Première passe : compter le nombre de lignes
with open(input_file, "r", encoding="utf-8") as f:
    n_lines = sum(1 for _ in f)

lines_per_file = n_lines // n_splits
extra = n_lines % n_splits  # lignes restantes

# Deuxième passe : écrire les fichiers
with open(input_file, "r", encoding="utf-8") as f:
    for i in range(n_splits):
        print(i)
        out_file = SMALLER_DATASET_PATH / f"metadata_{i+1}.jsonl"
        n_write = lines_per_file + (1 if i < extra else 0)  # répartir le reste sur les premiers fichiers
        with open(out_file, "w", encoding="utf-8") as fout:
            for _ in range(n_write):
                line = f.readline()
                if not line:
                    break
                fout.write(line)

print(f"Fichier divisé en {n_splits} parties")