import pandas as pd
import spacy
import re


# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def preprocess_text(text):
    """
    Preprocesses text by normalizing quotes and preparing for further processing.
    """
    # Normalize quotes
    text = re.sub(r"[‘’`]", r"'", text)
    text = re.sub(r"[“”]", r'"', text)

    return text

# Lemmatization function
def lemmatize_text(text):
    text = preprocess_text(text)
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Path to your Excel file
input_file = "../posts_first_targil.xlsx"

# Read the Excel file with multiple sheets
excel_data = pd.read_excel(input_file, sheet_name=None)  # Load all sheets as a dictionary of DataFrames

# Process each sheet
processed_sheets = {}
for sheet_name, df in excel_data.items():
    # Apply lemmatization to all string columns in the DataFrame
    lemmatized_df = df.applymap(lemmatize_text)
    processed_sheets[sheet_name] = lemmatized_df

# Save each lemmatized sheet to a separate Excel file
output_file = "output_files/lemmatized_file.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, lemmatized_df in processed_sheets.items():
        lemmatized_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Lemmatized Excel file saved as: {output_file}")
