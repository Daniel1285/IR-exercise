import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk

# Download NLTK resources if needed
nltk.download("punkt")

# File path to source documents
source_file = "../posts_first_targil.xlsx"
output_file = "output_files/sbert_vectors.csv"

# Load source documents
df = pd.read_excel(source_file, sheet_name=None)
if "J-P" in df:
    df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)

# Load pre-trained SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare results with Sheet and RowIndex
results = []

for sheet_name, data in df.items():
    for index, row in data.iterrows():
        if sheet_name == "A-J":
             combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
             combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))
        # Generate SBERT vector
        vector = model.encode(combined_text).tolist()
        # Append results
        results.append([sheet_name, index] + vector)

# Save results to CSV
header = ["Sheet", "RowIndex"] + [f"Dim{i}" for i in range(len(results[0]) - 2)]

with open(output_file, "w", encoding="utf-8") as file:
    file.write(",".join(header) + "\n")
    for row in results:
        file.write(",".join(map(str, row)) + "\n")

print(f"SBERT vectors saved to {output_file}")
