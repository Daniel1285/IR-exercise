import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK resources if not already done
nltk.download("punkt")

# File path to source Excel file
source_file = "../posts_first_targil.xlsx"
output_file = "../output_files/bert_vectors.csv"

# Load source documents
df = pd.read_excel(source_file, sheet_name=None)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    return word_tokenize(text)

# Function to generate BERT vectors
def get_bert_vector(text):
    # Tokenize and encode
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():  # Disable gradient computation
        outputs = model(**inputs)
    # Use the CLS token as the document vector
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
    return cls_embedding

# Prepare results
results = []

for sheet_name, data in df.items():
    for index, row in data.iterrows():
        if sheet_name == "A-J":
             combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
             combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))
        # Preprocess and generate vector
        bert_vector = get_bert_vector(combined_text)
        results.append([f"{sheet_name}_{index}"] + bert_vector.tolist())

# Save vectors to CSV
header = ["Sheet", "RowIndex"] + [f"Dim{i}" for i in range(bert_vector.shape[0])]
with open(output_file, "w", encoding="utf-8") as file:
    file.write(",".join(header) + "\n")
    for row in results:
        file.write(",".join(map(str, row)) + "\n")


print(f"BERT vectors with RowIndex saved to {output_file}")
