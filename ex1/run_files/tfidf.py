# import pandas as pd
# from rank_bm25 import BM25Okapi
# import csv
#
# # File paths
# input_file = "output_files/lemma_file.xlsx"  # Replace with the actual path to your Excel file
# output_file = "bm25_vectors.csv"
# df = pd.read_excel(input_file, sheet_name=None)
#
# excel_file = pd.ExcelFile(input_file)
# results = []
#
# if "J-P" in df:
#     df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)
#
#
# # Process each sheet
# for sheet_name, data in df.items():
#
#     # Initialize BM25
#     bm25 = BM25Okapi(data)
#
#     # Calculate vectors for each row
#     for index, row in data.iterrows():
#         scores = bm25.get_scores(row)
#         results.append([sheet_name, index] + scores.tolist())
#
# # Save results to a CSV file
# header = ["Sheet", "RowIndex"] + [f"Doc{i}" for i in range(len(results[0]) - 2)]
# with open(output_file, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     writer.writerows(results)
#
# print(f"BM25 vectors saved to {output_file}")
#
#
#
from scipy.sparse import load_npz

# Load the sparse matrix
sparse_matrix = load_npz("../bm25/lemma/BBC.npz")

# Inspect the sparse matrix
print("Sparse matrix shape:", sparse_matrix.shape)
print("Non-zero elements:", sparse_matrix.nnz)
print("Matrix contents:")
print(sparse_matrix.toarray())  # Convert to dense matrix for inspection