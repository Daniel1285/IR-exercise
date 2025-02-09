# README - Sentiment Analysis of Articles

## Student Details

- **Student Name:** Daniel Shalom Cohen
- **Student Name:** Natan Stern

## Project Description

The project aims to analyze the sentiment of news articles, categorizing them into two main groups:

- **Israeli** - Articles with a positive bias towards Israel.
- **Palestinian** - Articles with a positive bias towards Palestinians.

The analysis is based on predefined keywords associated with each category, and an attempt was made to classify articles accordingly.

## Methodology

1. **Data Collection**: Loading an Excel file containing the articles.
2. **Text Processing**:
   - Combining titles, subtitles, and body text (depending on the dataset structure).
   - Splitting texts into sentences using punctuation markers (e.g., periods, question marks, exclamation marks).
3. **Sentiment Identification**:
   - Searching for keywords from predefined categories (Pro-Israeli and Pro-Palestinian).
   - Assigning each sentence to the appropriate category based on the keywords it contains.

## Choice of Method

We chose a keyword-based approach because it allows for a quick understanding of article biases without requiring complex Natural Language Processing (NLP) models. Additionally, this approach is easy to implement and does not require pre-labeled data.

### Challenges Encountered

- **Correct Sentiment Identification**: Some sentences contained words from both categories, causing classification issues.
- **Unexpected Biases**: Some articles were written sarcastically or used words from a specific category in a way that did not necessarily indicate a positive sentiment.
- **Category Imbalance**: Some articles had a significantly higher number of sentences with a particular sentiment, potentially skewing the overall analysis.

## Conclusions

1. **The current method provides initial insights but lacks precision** - Using deep learning models or more complex semantic analysis could improve accuracy.
2. **Certain newspapers exhibit noticeable bias trends** - Differences between sources can be observed.
3. **If we were newspaper editors, we would advise journalists to be aware of how their phrasing might be perceived as political bias.**

## Results

paper: A-J label: POS-P score: 0.7497990552337218

paper: BBC label: POS-P score: 0.7481500840890505

paper: J-P label: POS-P score: 0.7474796354869099

paper: NY-T label: POS-P score: 0.744396989673225

### Error Analysis and Insights

- **Common Errors**: Because the sentiment is not perfect, some articles were misclassified because certain words appeared outside their political context.



## Summary

Despite the challenges, the analysis reveals clear trends and provides a basic understanding of the sentiment tendencies in various articles. With further technological improvements, classification accuracy can be significantly enhanced.

