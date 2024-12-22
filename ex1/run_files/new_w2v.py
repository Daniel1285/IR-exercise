import pandas as pd
import string
import re
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import csv
import nltk

nltk.download('punkt')
nltk.download('stopwords')


input_file = "../output_files/lemma_file.xlsx"
output_file = "../output_files/new_w2v_lemma_vectors_2.csv"

df = pd.read_excel(input_file, sheet_name=None)

if "J-P" in df:
    df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)

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
                idf_value = idf_dict[word]
                weighted_vector = vector * idf_value  # Multiply word vector by IDF
                results.append([sheet_name, index, word] + weighted_vector.tolist())


# Save results to a CSV file
header = ["Sheet", "RowIndex", "Word"] + [f"Dim{i}" for i in range(glove_model.vector_size)]
with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results)

print(f"Word vectors saved to {output_file}")


# File path
output_file = "../output_files/new_w2v_lemma_vectors_2.csv"

try:
    # Read the first 10 rows of the CSV file
    data = pd.read_csv(output_file, nrows=10)
    print(data)
except FileNotFoundError:
    print(f"File '{output_file}' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")




# Load the Word2Vec results into a DataFrame
input_file = "../output_files/new_w2v_lemma_vectors_2.csv"
output_file = "../output_files/new_word2vec_mean_vectors.csv"

# Load the word vectors
df = pd.read_csv(input_file)

# Group by Sheet and RowIndex and compute the mean for each dimension
dim_columns = [col for col in df.columns if col.startswith("Dim")]
doc_vectors = (
    df.groupby(["Sheet", "RowIndex"])[dim_columns]
    .mean()
    .reset_index()
)
doc_vectors.to_csv(output_file, index=False)

print(f"Averaged document vectors saved to {output_file}")

output_file = "../output_files/new_word2vec_mean_vectors.csv"

try:
    data = pd.read_csv(output_file)
    print(data.head())
except FileNotFoundError:
    print(f"File '{output_file}' not found. Please check the file path.")
