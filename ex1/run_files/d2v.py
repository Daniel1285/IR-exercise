import pandas as pd
import re
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk

# Download NLTK resources if not already downloaded
nltk.download('punkt')

# File path to the source Excel file
source_file = "../posts_first_targil.xlsx"  # Replace with your actual file path

# Load source Excel file
df = pd.read_excel(source_file, sheet_name=None)
if "J-P" in df:
    df["J-P"].rename(columns={"Body": "Body Text"}, inplace=True)


# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return word_tokenize(text)  # Tokenize the text

# Prepare TaggedDocuments
tagged_documents = []
for sheet_name, data in df.items():
    for index, row in data.iterrows():
        if sheet_name == "A-J":
            combined_text = " ".join(str(row[col]) for col in ['title', 'sub_title', 'Body Text'] if pd.notna(row[col]))
        else:
            combined_text = " ".join(str(row[col]) for col in ['title', 'Body Text'] if pd.notna(row[col]))

        tokens = preprocess_text(combined_text)
        tagged_documents.append(TaggedDocument(words=tokens, tags=[f"{sheet_name}_{index}"]))

# Train Doc2Vec model
model = Doc2Vec(vector_size=300, min_count=2, epochs=40, workers=4)
model.build_vocab(tagged_documents)
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

# Save document vectors to CSV
output_file = "../output_files/doc2vec_vectors.csv"
header = "Sheet,RowIndex," + ",".join([f"Dim{i}" for i in range(model.vector_size)])
with open(output_file, "w", encoding="utf-8") as file:
    file.write(header + "\n")
    for doc_id, doc in enumerate(tagged_documents):
        # Extract sheet name and row index from doc.tags[0]
        sheet, row_index = doc.tags[0].split("_")
        vector = model.dv[doc.tags[0]].tolist()
        file.write(f"{sheet},{row_index}," + ",".join(map(str, vector)) + "\n")

print(f"Document vectors with RowIndex saved to {output_file}")
