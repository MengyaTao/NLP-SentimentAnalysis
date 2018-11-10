from __future__ import division
# import the packages
import csv
import nltk
# nltk.download()
# download corporus "wordnet", "stopwords"
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# read in the data of abstract and score
def input_file(file):
    raw = open(file, 'rU')
    data = csv.reader(raw)
    abstract = []
    score = []
    for row in data:
        abstract.append(row[0])
        score.append(row[1])
    results = zip(abstract, score)
    return results

#===============================================================================
# tokenize the paragraph
# Normalization steps: 1) put every character into lower case for normalization
# 2) remove stopwords 3)stemming words (a way for further text normalization)  
#===============================================================================
def abstract_to_words (paragraph):
    # Function to convert a raw paragraph of abstract to a string of words
    # The input is a single string
    # The output is a single string 
    #
    # Assume, the order of data cleaning processes doesn't matter
    # 0. remove numbers
    # 1. tokenize the string into words 
    # 2. convert words to lower case
    # 3. remove the stop words
    # 4. stemming the words/lemmatization (used lemmatizated results)
    # 5. remove punctuation
    # 6. join the words back into one string separate by space

    letters_only = re.sub("[^a-zA-Z]", " ", paragraph)
    para_no_pun = letters_only.translate(None, string.punctuation)
    tokens = para_no_pun.lower().split()
    # tokens = word_tokenize(para_no_pun)

    words_low = [w.lower() for w in tokens]
    stop_words = set(stopwords.words('english'))
    meaningful_words = [word for word in words_low if not word in stop_words]
    
    # porter = nltk.PorterStemmer()
    wnl = nltk.WordNetLemmatizer()
    # words_stem = [porter.stem(t) for t in meaningful_words]
    words_lemma = [wnl.lemmatize(t) for t in meaningful_words]
    words = " ".join(words_lemma)
    return words

def data_clean_output (input_data_file, output_data_name):
    data = input_file(input_data_file)
    abs_clean = []
    score = []
    for m in range(len(data)):
        abs_clean.append(abstract_to_words(data[m][0]))
        score.append(data[m][1])
    rows = zip(abs_clean, score)
    
    with open(output_data_name + '.csv', 'wb') as file:
        writer = csv.writer(file)
        writer.writerow(('Abstract', 'Score'))
        for row in rows:
            writer.writerow(row)
    file.close()
            

data_clean_output("WOS_data.csv", "data_clean_lemma3")
data_clean_output("training_data_2500.csv", "training_data_clean")
data_clean_output("test_data_480.csv", "test_data_clean")
data_clean_output("validation_data_500.csv", "validation_data_clean")

