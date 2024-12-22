import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import nltk
nltk.download("punkt")

source_file = "../posts_first_targil.xlsx"
output_file = "../output_files/new_bert_vectors.csv"

df = pd.read_excel(source_file, sheet_name=None)
if "J-P" in df:
    df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def calculate_idf(corpus):
    vectorizer = TfidfVectorizer(use_idf=True, stop_words="english")
    vectorizer.fit(corpus)
    idf_dict = defaultdict(lambda: 0)
    for word, idf in zip(vectorizer.get_feature_names_out(), vectorizer.idf_):
        idf_dict[word] = idf
    return idf_dict

corpus = []
for sheet_name, data in df.items():
    for index, row in data.iterrows():
        if sheet_name == 'A-J':
            combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
            combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))
        corpus.append(combined_text)

idf_dict = calculate_idf(corpus)


def get_bert_vectors(text_chunk):
    inputs = tokenizer(text_chunk, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [sequence_length, hidden_size]
    attention_mask = inputs["attention_mask"].squeeze(0)  # Shape: [sequence_length]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))  # List of tokens
    print(tokens)
    exit()
    return tokens, token_embeddings, attention_mask

# Function to process subwords into full word embeddings
def process_tokens(tokens, token_embeddings, attention_mask, idf_dict):
    word_embeddings = []
    current_word = ""
    current_word_vectors = []

    for token, embedding, mask in zip(tokens, token_embeddings, attention_mask):
        if mask == 0 or token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):  # Subword continuation
            current_word += token[2:]
            current_word_vectors.append(embedding)
        else:  # New word starts
            if current_word:  # Combine previous word embeddings
                combined_embedding = torch.mean(torch.stack(current_word_vectors), dim=0)
                idf = idf_dict.get(current_word, 1.0)  # Default IDF to 1.0 if not found
                word_embeddings.append(combined_embedding * idf)

            # Start new word
            current_word = token
            current_word_vectors = [embedding]

    # Process the last word
    if current_word:
        combined_embedding = torch.mean(torch.stack(current_word_vectors), dim=0)
        idf = idf_dict.get(current_word, 1.0)
        word_embeddings.append(combined_embedding * idf)

    return word_embeddings
# Function to process an entire document
def process_document(text, idf_dict):
    tokens = tokenizer.tokenize(text)
    max_tokens = 512
    num_chunks = (len(tokens) + max_tokens - 1) // max_tokens  # Ceiling division
    all_word_embeddings = []

    for i in range(num_chunks):
        chunk_tokens = tokens[i * max_tokens : (i + 1) * max_tokens]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        tokens, embeddings, attention_mask = get_bert_vectors(chunk_text)
        chunk_word_embeddings = process_tokens(tokens, embeddings, attention_mask, idf_dict)
        all_word_embeddings.extend(chunk_word_embeddings)

    # Aggregate all word embeddings for the document (e.g., by mean or sum)
    document_embedding = torch.mean(torch.stack(all_word_embeddings), dim=0)
    return document_embedding


results = []
for sheet_name, data in df.items():
    for index, row in data.iterrows():
        if sheet_name == "A-J":
            combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
            combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))

        # Generate BERT vectors for the document
        bert_vector = process_document(combined_text, idf_dict)
        vector_list = bert_vector.tolist()
        results.append([sheet_name, index] + vector_list)
        print(vector_list)



# Save vectors to CSV
header = ["Sheet", "RowIndex"] + [f"Dim{i}" for i in range(bert_vector.shape[0])]
with open(output_file, "w", encoding="utf-8") as file:
    file.write(",".join(header) + "\n")
    for row in results:
        file.write(",".join(map(str, row)) + "\n")

print(f"BERT vectors with RowIndex saved to {output_file}")


output_file = "../output_files/new_bert_vectors.csv"

try:
    data = pd.read_csv(output_file)
    print(data.head())
except FileNotFoundError:
    print(f"File '{output_file}' not found. Please check the file path.")