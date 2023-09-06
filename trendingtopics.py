import pandas as pd
import re
import os
import argparse
import sys

import PyPDF2

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

"""
read_pdf
---------------------
params: 
    file_path (str): a valid path to a pdf containing podcast summaries
returns:
    full_text (str): The text in a pdf in a concatenated string
"""
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.getNumPages()
        full_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            full_text += page.extractText()
        return full_text

"""
Error handling to ensure that valid pdf files are passed in as arguments, load them into
a concatenated string
"""

if len(sys.argv) < 2:
    raise ValueError("Usage: python script.py <pdf_file1> <pdf_file2> ...")

# Extract text from each PDF and concatenate into a single string
concatenated_text = ""
for pdf_file in sys.argv[1:]:
    if not os.path.isfile(pdf_file):
        raise FileNotFoundError(f"Error: '{pdf_file}' is not a valid file.")
    if not pdf_file.lower().endswith(".pdf"):
        raise ValueError(f"Error: '{pdf_file}' is not a PDF file.")

    text_from_pdf = read_pdf(pdf_file)
    concatenated_text += text_from_pdf

"""
Performing LDA on Script to extract 

"""
podcasts = concatenated_text.split('\n')

# Tokenize and clean text
stop_words = set(stopwords.words('english'))
podcasts_cleaned = []
for text in podcasts:
    word_tokens = word_tokenize(text)
    filtered_text = [w.lower() for w in word_tokens if not w in stop_words and w.isalpha()]
    podcasts_cleaned.append(filtered_text)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(podcasts_cleaned)

# Create a bag-of-words model for each document i.e for each document we create a dictionary reporting how many
# words and how many times those words appear
corpus = [dictionary.doc2bow(podcast) for podcast in podcasts_cleaned]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

# Print the topics
for idx in range(10):
    # Print the first 10 most representative topics
    print("Topic #%s:" % idx, lda_model.print_topic(idx, 10))
    
