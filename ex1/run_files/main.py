import pandas as pd
import re

# Your clean_text function
def clean_text(text):
    regx = r"((?<!\w)[^\s\w]|[^\s\w](?!\w))"
    clean_t = re.sub(regx, r" \1 ", text)
    # Replace multiple spaces with a single space
    return re.sub(r"\s+", " ", clean_t).strip()


input_file = "../posts_first_targil.xlsx"

# Read the Excel file with multiple sheets
excel_data = pd.read_excel(input_file, sheet_name=None)  # Load all sheets as a dictionary of DataFrames

# Process each sheet
processed_sheets = {}
for sheet_name, df in excel_data.items():
    # Apply clean_text to all string columns in the DataFrame
    processed_df = df.applymap(clean_text)
    processed_sheets[sheet_name] = processed_df

# Save each processed sheet to a separate Excel file
output_file = "output_files/processed_file.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, processed_df in processed_sheets.items():
        processed_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Processed Excel file saved as: {output_file}")



