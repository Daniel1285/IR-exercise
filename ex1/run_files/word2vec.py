import pandas as pd
import numpy as np
import string
import re
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.data import find
import csv

# Download NLTK resources (ensure you have nltk installed: pip install nltk)
import nltk

nltk.download('punkt')
nltk.download('stopwords')



# File paths
input_file = "../output_files/lemma_file.xlsx"  # Replace with your Excel file path
output_file = "w2v_lemma_vectors_2.csv"

df = pd.read_excel(input_file, sheet_name=None)

if "J-P" in df:
    df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)

# Load GloVe vectors via gensim downloader
try:
    glove_model = api.load("glove-wiki-gigaword-300")  # 300-dimensional GloVe vectors
except Exception as e:
    print(f"Error loading model: {e}")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove digits and dates
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]  # Remove stopwords


# Process each sheet
results = []

for sheet_name, data in df.items():
    for index, row in data.iterrows():
        # Combine text from relevant columns
        if sheet_name == 'A-J':
            combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
            combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))

        # Preprocess text and get tokens
        tokens = preprocess_text(combined_text)

        # Extract vectors for each word
        for word in tokens:
            if word in glove_model:
                vector = glove_model[word]
                results.append([sheet_name, index, word] + vector.tolist())

# Save results to a CSV file
header = ["Sheet", "RowIndex", "Word"] + [f"Dim{i}" for i in range(glove_model.vector_size)]
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results)

print(f"Word vectors saved to {output_file}")


# File path
output_file = "../output_files/w2v_clean_vectors.csv"  # Replace with your file path

try:
    # Read the first 10 rows of the CSV file
    data = pd.read_csv(output_file, nrows=10)
    print(data)
except FileNotFoundError:
    print(f"File '{output_file}' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
