import os
import json
import pandas as pd
import numpy as np


def calculate_feature_importance(input_dir, output_file):
    """
    Calculate feature importance using variance and mean TF-IDF for each matrix.
    Save results as an Excel file with proper feature names.
    """
    writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

    for matrix_file in os.listdir(input_dir):
        if matrix_file.endswith(".json"):
            file_path = os.path.join(input_dir, matrix_file)

            # Load TF-IDF matrix from JSON
            with open(file_path, "r") as f:
                tfidf_data = json.load(f)

            # Convert TF-IDF data to a DataFrame
            tfidf_df = pd.DataFrame.from_dict(tfidf_data, orient="index")
            print(tfidf_df.info)
            print(tfidf_df.describe())
            exit()

            # Ensure columns have proper names
            tfidf_df.columns.name = "Feature"

            # Compute Variance and Mean TF-IDF
            variances = tfidf_df.var(axis=0)
            mean_tfidf = tfidf_df.mean(axis=0)

            # Create DataFrames for each metric
            variance_df = pd.DataFrame({
                "Feature": variances.index,  # Correctly get feature names
                "Variance": variances.values
            }).sort_values(by="Variance", ascending=False)

            print(variance_df)
            print("==========================================")
            print("==========================================")
            print("==========================================")
            exit()

            mean_tfidf_df = pd.DataFrame({
                "Feature": mean_tfidf.index,  # Correctly get feature names
                "Mean TF-IDF": mean_tfidf.values
            }).sort_values(by="Mean TF-IDF", ascending=False)

            # Write to Excel as separate sheets
            sheet_name = os.path.splitext(matrix_file)[0]
            variance_df.to_excel(writer, sheet_name=f"{sheet_name}_Variance", index=False)
            mean_tfidf_df.to_excel(writer, sheet_name=f"{sheet_name}_MeanTFIDF", index=False)

    # Save the Excel file
    writer._save()
    print(f"Feature importance saved to {output_file}")



if __name__ == "__main__":

    input_directory = "output_files/tfidf_output_word"  # Replace with your directory
    output_excel = "TFIDF_Feature_Importance_words.xlsx"
    calculate_feature_importance(input_directory, output_excel)

    input_directory = "output_files/tfidf_output_lemma"  # Replace with your directory
    output_excel = "TFIDF_Feature_Importance_lemma.xlsx"
    calculate_feature_importance(input_directory, output_excel)
