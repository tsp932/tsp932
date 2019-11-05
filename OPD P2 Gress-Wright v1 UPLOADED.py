# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:52:07 2019

@author: Jessica
"""
### PORTFOLIO 2 
## Instructions
# For this assignment you will:
#   1. Open, import, and/or read a text collection
#   2. Pre-process and describe your collection
#   3. Select articles using a query
#   4. Model and visualize the topics in your subset

# STYLE NOTE:  in the comments a "" refers to code, while a '' is a string.

###   1. OK Open, import, and/or read a text collection
#   DOWNLOAD GUARDIAN'S TEXT DATASET USING GUARDIAN'S API
#   JGW user key    b37a10ff-7282-40bf-a60f-bacb07a22cd7
#   Script to use to download datafile (with parameters for key, dates, etc. filled in).
#   Script is derived from the Guardian's API documentation page (saved as .pdf)
import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta
# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)
# Sample URL

# http://content.guardianapis.com/search?from-date=2016-01-02&
# to-date=2016-01-02&order-by=newest&show-fields=all&page-size=200
# &api-key=your-api-key-goes-here
# Change this for your API key:
MY_API_KEY = 'b37a10ff-7282-40bf-a60f-bacb07a22cd7'
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
        'from-date': "", # leave empty, change start_date / end_date variables instead
        'to-date': "",
        'order-by': "newest",
        'show-fields': 'all',
        'page-size': 200,
        'api-key': MY_API_KEY
}
# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
# Update these dates to suit your own needs.
start_date = date(2019, 1, 1)
end_date = date(2019, 10, 28)
dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        # then let's download it
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']
#
        with open(fname, 'w') as f:
            print("Writing to", fname)
#       Saving the actual article files to disk ("w" means write)
            # re-serialize it for pretty indentation
            f.write(json.dumps(all_results, indent=2))
# READING YOUR JSON FILES
#   Given that you've run the above script and managed to import a series of json 
#       files, here's how you read them all into two variables: a list of ids 
#       and a list of plain texts.
import json
import os
# "os means operating system.  We need this because we need to be able to read 
#   data from the net and then save to own hard disk with a file name 
# Update to the directory that contains your json files
# Note the trailing "/"
directory_name = "theguardian/collection/"
# this is creating a file path on the hard disk to a folder named "theguardian"
#   and a subfolder named "collection"
ids = list()
guard_texts = list()
# This creates two empty lists:  "ids" and "texts".  Now we can begin to work with them.
# First we create a for loop to load the data (articles and associated id's for time period)
for filename in os.listdir(directory_name):
# We now go to the folder and read each file of the type "json".
    if filename.endswith(".json"):
        # This downloading pro
        with open(directory_name + filename) as json_file:
            
            loopdata = json.load(json_file)
            # The data variable "loopdata" is created in Python with the content from 
            # the datafile on the hard disk-- for the purposes of reading the Guardian's 
            # .json files into the program and stored in the program's memory 
            # so it can be worked with.            
            # This is done by means of the following "for loop", which reads the 
            # respective id's and text for each article.  The file now becomes 
            # an appended list. 
            
        for article in loopdata:
                id = article['id']
                fields = article['fields']
                doc_type = article['type']
                section_id = article['sectionId']
                section_name = article['sectionName']
                if section_name == 'Opinion':
                    # only artilce from the opnion asection are appended to texts 
                    text = fields['bodyText'] if fields['bodyText'] else ""
                    # if the field is labelled "body text" then it is appended to the 
                    # "texts" list, if not, then a blank is appended
                    ids.append(id)
                    guard_texts.append(text)
                # N.B "append" is okay to use (rather than "extend") because each item 
                # (i.e. article) is a single item. If each article were instead a list, 
                # and you wanted each item in that article list appended one by one,
                # you would have to use "extend" not "append"
                
## INITIAL EXPLORATION OF THE DATA                
print("Number of ids: %d" % len(ids))
print("Number of texts: %d" % len(guard_texts))
# The list has 64915 indices ("id's") i.e 64915 texts
#   TEST PRINT a selected article
# print("Article 100 reads as follows:  ",texts[100]) #why "texts" plural and not "text"?

###  "2. PRE-PROCESS AND DESCRIBE YOUR COLLECTION
#   "Derive a document-term matrix for your collection. Explain which
#   pre-processing steps you take, for example do you: limit the vocabulary size,
#   apply a token pattern, remove stopwords, or use tf-idf weighting?
#   Describe the characteristics of your collection. How many documents, what is
#   the average length of the documents, how many terms before/after
#   pre-processing, how much words in total, etc. Show that you’re getting “good”
#   terms for further analysis."

#       2.0 OVERVIEW PREPROCESSING 
#       2.1 import libraries and toolkits with which to process and analyze the data
#       2.2 remove stop words using nltk (English)
#       2.3 import CountVectorizer model
#       2.4 limit the vocabulary size to 10,000 terms
#       2.5 apply a token pattern
#       2.6 fit and transform the data to create a document-term matrix with term counts

##  2.1 In order to begin the pre-processing, later analysis and evt. visualization, import following
import sklearn
import numpy as np
import pandas as pd
import matplotlib
from pandas import DataFrame 
##  2.2 Remove stop words (use nltk’s stopwords)
#           Stop words removes many of frequent but insignificant terms common to all texts, 
#           such as "of" "the" "but".  This helps keep a focus on truly significant words
#           N.B. "stop words" fram nltk is a better version of stopwords than sklearn's. 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as sw
#           "sw" now contains direction as to where to find the stop words on the hard disk:
#                <WordListCorpusReader in 'C:\\Users\\Jessica\\AppData\\Roaming\
#               \nltk_data\\corpora\\stopwords'>

##  2.3 Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
model_vect = CountVectorizer ()
#   Initialize CountVectorizer, i.e. create a copy of the "tool box" called 
#       "model_vect" with empty parameters (i.e. no data yet, just creating an 
#       empty container but with working machinery)

##  2.4 Limit the vocabulary to 10,000 unique terms
#       AND
##  2.5 Apply a token_pattern

model_vect = CountVectorizer(max_features=10000, stop_words=sw.words('english'), token_pattern=r'[a-zA-Z\-][a-zA-Z\-]{2,}')
#           We specify the contents of the tool box, i.e. the parameters of model_vect.
#           We limit the vocabulary to 10000 terms (i.e. "features", "tokens", "words")
#           We specify our token pattern as "r'[a-zA-Z\-][a-zA-Z\-]{2,}'"
#               '' means that the tokenizer matches strings 
#               "[]" specifies a character class, i.e. the set of characters,.
#               which count as a match, e.g. [a-zA-Z\-][a-zA-Z\-] 
#               "a-zA-Z\-" means any letter, upper or lower case, and/or hyphen character.  
#                  Words separated by a hyphen count as one token or term.
#                   N.B. The backslash "\" is a regular expression symbol which  
#                   excepts the "-" from being interpreted as a meta-character.
# 
#               {2,} means that a token must have at least two repetitions of the pattern, 
#                   so a token must consist of at least two characters
#               The blank after "," means there is no upper character limit for a token
model_vect
#   Tests output.  List of stop-words looks right
#       CountVectorizer(analyzer='word', binary=False, decode_error='strict',
#               dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
#                lowercase=True, max_df=1.0, max_features=10000, min_df=1,
#               ngram_range=(1, 1), preprocessor=None,
#                stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
#                           'ourselves', 'you', "you're", "you've", "you'll",
#                           "you'd", 'your', 'yours', 'yourself', 'yourselves',
#                           'he', 'him', 'his', 'himself', 'she', "she's",
#                           'her', 'hers', 'herself', 'it', "it's", 'its',
#                            'itself', ...],
#                strip_accents=None, token_pattern='[a-zA-Z\\-][a-zA-Z\\-]{2,}',
#                tokenizer=None, vocabulary=None)

## 2.6 Fit and transform the data into a document term matrix

guard_vect = model_vect.fit_transform(guard_texts)
#       The Count_Vectorizer model is now trained on the "texts" dataset and analysis may begin
#       The data has both been fitted and transformed using ".fit_transform". 
#       We can fit and transform at same time because we will not be adding more 
#       data later--this is all the data we are going to be working with.

##  2.7 Describe the resulting matrix and contents

#   2.7.1 Describing the matrix as an object
guard_vect
#  "guard_vect" when written in the console itself, returns the shape and class:
#   <64915x10000 sparse matrix of type '<class 'numpy.int64'>'
#	with 16456996 stored elements in Compressed Sparse Row format>:
#   That means 64915 rows, 10000 columns and 16,456,996 cell values (INCLUDING ZEROES???).

#   Alternatively, one can write
print('Shape: (%i, %i)' % guard_vect.shape)
guard_vect.shape

#   NOTE: writing the string as 'Shape (%i, %i)' is a way
#       to insert the resulting values from the command "guard_vect.shape" in the 
#       string itself.  The "i" means the result should be expressed as a "signed
#       integer decimal" i.e. could be a postive or negative integer in base 10.
# 
 ## 2.7.2 Printing the matrix
print(guard_vect)

#   I printed the matrix just to see what I got.  This is an enormous matrix   
#   presented like a dictionary.  A sparse matrix, it consists of the co-ordinates 
#   of a cell, the "rows" being the document indices and the "columns" being the term 
#   indices.  The value in the cell is the count for that term in that document.  
#   
#   As an example, the first item in the "dictionary" is "(0, 667) 1".  This means 
#   that the program began by examining the first article (i.e. document) in the 
#   collection, with index 0.  And, out of the 10,000 terms in total, which 
#   CountVectorizer had calculated, the first one it came across in that document
#   was the term 667, which occurred 1 time in this document.   
#
#   The CountVectorizer's term counts are per document, per term for a reason.
#   This is essential to comparing the "fingerprints" of each term in a collection, 
#   as spread across documents, as well as to comparing fingerprints of documents
#   as spread across terms.  It is also the basis for calculating term weights, 
#   document vectors, and cosine similarities.

#   A further point to note is that there are no zero values.  That is because this 
#   matrix is a sparse matrix, i.e. all the zero counts, are left out to save memory.
#   We can see that Compressed Sparse row format returns only the
#   cells with numbers in them.  If we ask for a cell which is not represented 
#   a 0 is returned. 
#
#   Converting to a 2-D matrix is only possible if the collection is small.  This
#   sample is too big, over 5000 documents, so below we will take a slice of the 
#   collection and then use pandas to produce a 2D document-term array.
#   
#   In that connection, using pd.dataframe().groupby() may be useful - some advise to create a list first
        

## 2.9 OK Take a look at the terms for a random sample. Do they make sense?
import random
random.sample(model_vect.get_feature_names(), 10)
#   Yes, they do make sense. Code returned 10 random terms: 
#       'sparked','licence','adopting','sketch','shapes','billionaire','stretches',
#       'angela', 'lab',  'wholly'
#       
## 2.10 Bonus: The resulting matrix is a sparse matrix. What is the sparsity of this matrix? OK

dense_count = np.prod(guard_vect.shape)
zeros_count = dense_count - guard_vect.size
zeros_count / dense_count
print(zeros_count / dense_count)
#   NOTE:  The calculation of sparsity requires comparing to the dense matrix which does include zero cell values.  
#       That can only be done in numpy.  Therefore we must use numpy functions.
#       numpy has a function '.prod' which can be applied to guard_vect.shape.  
#       That gives us our dense matrix count.
# (((RESUME 2. DESCRIBE DATA)))
#       Why guard_vect.shape and not just guard_vect?? Don't know
#   NOTE:  Count_Vectorizer, on the other hand, has a function ".size".  a) 'guard_vect.size.' clearly gives the sparse count, which means that this information
#       must be taken from Count_vectorizer, not numpy, because C_v, not numpy, generates sparse returns.  
#       This means that both Count_vectorizer and numpy remain loaded and "alive".  But to use numpy, you must "call" it e.g. np.prod
#       How would one know this?? I guess these functions np ".prod" and C_v ".size" are found through Google or help

##  2.11 How long are the texts? 
#           2.11.1 Calculate the total number of characters in the data.
#                  Use summation pattern
total = 0
for text in guard_texts:
    total = total + len(text)
# cannot do len(guard_vect) but the number of documents i 5531  len(guard_texts))
total_chars=(total/len(guard_texts))
print (total_chars)


# SHOULD WE BE CREATING A VOCABULARY??
#   {} is a sign for a vocabulary, a kind of dictionary object.  A vocabulary is 
#   a list of tuples consisting each of a key and a value.  Here, the key is the index number (i.e. id number for 
#   the term) and the value is the term (i.e. the token (or word) associated with that key).
#   This vocabulary is the basis of all subsequent analysis, e.g. term counts, 
#   weight counts, document vectors, cosine simililarities, and topic modelling

### 3. Select articles using a query
#SHOW PART of the document-term count matrix

##   3.1 Give the indexes of the top-10 most used words.
#       3.1.1  To do this, one must first identify the 10 most used words (terms).
#       However, existing term counts are just one document.  
#       So, to get the top-10 most used words in the whole collection, one must first 
#           get each term's count in the whoe collection, i.e. sum the individual 
#           document counts for a given term  
#       As each column represents one term, it is necessary to sum down all the rows of 
#           a given column to get the collection count for that term
#       Summing each and every column will result in an array of one row and 10,000 columns. 
#       To sum each column, use numpy's "sum" function
#   You want the count for all the docs per term
#   A1 function means array 1 e.g. says ditch the one dimensiont s
#  

sums_by_term = guard_vect.sum(axis=0).A1
top_idxs = (-sums_by_term).argsort()[:10]
top_idxs
#       Testing what happens when slice is enlarged to 20 columns 
top_idxs = (-sums_by_term).argsort()[:20]
print(top_idxs)
top_idxs = (-sums_by_term).argsort()[:10]
# Now back to 10 terms
#       How the code works:  
#       We start with the existing guard_vect matrix. We use the ".sum" function
#           to sum the down the cells in each column. 
#       We ensure this, first, by specifying the "0" axis.  This "flattens" the array 
#           by collapsing the 0 axis, which is all the rows, into just one row, 
#           retaining the 173.000 columns
#       The attribute ".A1" converts the 1 dimensional matrix to an array 
#           (the reason to convert is that in matrix you cannot sort, in array you can sort)
#       Once all the term sums are in an array, you can sort it using ".argsort
#           sorts the columns according to alphanumeric CELL value: A-Z, then a-z, then 0-9.  
#           This order of sorting means that numbers are sorted in ASCENDING order, lowest to highest!
#       N.B.  The values which are returned are NOT the term count sums!! they are 
#           the "index" numbers of the respective columns, i.e. the column POSITIONS
#           identifying the 10 highest sums.  
#           So ".A1" does not really move the columns to new positions.  Instead
#           it makes a new list, containing ranked index numbers
#   
#       Moreover, we want a slice of just the top 10 most used words, in DESCENDING 
#           order, highest to lowest.  
#        
#       We ensure this by using the "-" to indicate reverse order, not lowest to highest,
#           but highest to lowest: "top_idxs = (-sums_by_term).argsort()[:10]" 
#           i.e. "to get the top indexes, start from the end and take 10 column 
#           values in reverse order (the last, the next to last, the next next 
#           to last, etc.).  
#       The result is a list of 10 values, which, again, are INDEXES.
#       
#       N.B. because the original term column postion IDs are retained we can now
#           find the respective term names and sums for each item in the list. 
#  
# (((RESUME 3 HERE))) 
#        
##  3.2 Give the WORDS belonging to the top-10 indexes.
# Tip: Tip: TO GET THE WORDS THAT MATCH THE INDICES Use an inverted vocabulary found 
#   in Count_vectorizer (see slides).
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: " , top_words)

##  3.3  Show the document vectors for 10 random documents. Limit the vector to
#        only the top-10 most used words.
# Note: To slice a sparse matrix use A[list1, :][:, list2] 
#       (see https://stackoverflow.com/questions/7609108/slicing-sparse-scipy-matrix).
# only the top-10 most used words.
# in the list 'ids' we have the 'ID' that the Guardian give each document 
# the for loop goes though the list of random numbers that range from
# 0 to the length(=number of) documents  we don't have any name for the
# document except the ids indetification so we print that
print(" random posts counts of 10 most frequent words")
import random
random_sample = random.sample(range(0,len(guard_texts)), 10)
print("Selection: random sample ")
for ii in random_sample:
     ci=ids[ii]
     print(ii, ",",ci," /  ")
##  3.4 Use pandas’ dataframe to display your selection and add column names 
#       (words) and row names (document indexes).
# a dense matrix is created with each random sample and the count of the 
# top 10 words in put in the matrix - bur first the actual 10 words are printed
print(' ')    
print("Selection of words: ")
print(top_words)
sub_matrix = guard_vect[random_sample, :][:, top_idxs].todense()
print (sub_matrix)
print(' ')
df=pd.DataFrame(columns=top_words, index=random_sample, data=sub_matrix )
print(df)

##  3.5 Reflections on the pros and cons of absolute word counts

#   Absolute word counts can contribute to misleading analyses when documents vary 
#   significantly in length.  One document may have 10x the references to women  
#   as another simply because it is 10x longer.  An absolute count is misleading, 
#   because the "fingerprint" i.e. the proportion of hits, is the same in both cases.
#           
#   What about this collection?  Comparing the length of cocuments may show something about 
#   how far we can trust absolute word counts in this case.

#   We test by first measuring the total number of words (including repetitions)    
#   in the collection.  Dividing by the number of documents gives us the mean number 
#   of words in each document.  To find the total each document is 'split' into words. 
#
#   But what about the difference in lengths between documents?  That is the important
#   consideration.  If there is a wide variation in length, it is more misleading 
#   to rely on absolute counts than if all the documents are the same length.
#   To find the shortest and longest documents, the highest and 
#   lowest word counts are kept as a running comparison, starting with the number of 
#   words in the first document.  To ensure that the for loop does keep going until
#   it finds the lowest count above zero, we code as follows:
  
total_words = 0
hi=len(guard_texts[0])
lo=hi
for text in guard_texts:
    words=text.split() 
    x=len(words)
    if x>hi:
        hi=x
    if (x<lo) and (x>0):
            lo=x
    total_words = total_words + x
#   as the loop executes the 'total_words' variable grows with the number of 
#   words in each document.  To find the average this total is divided by the 
#   tota number of documents . We cannot do len(guard_vect) but the 
#   number of documents is found by len(guard_texts))
average_words=(total_words/len(guard_texts))
print ("the average document contains ", average_words , "words") 
print ("the longest is ", hi, " words, the shortest is ", lo, " words" )

    
### 3.6 Weighting the document and term frequencies:  reflecting on and calculating TF-IDF 
#   
#   3.6.1 Why use weights?  
#       The reason is that we need to account for the fact that longer
#       documents have higher words counts.  Zipf's law states that the frequency of 
#       of a word in a collection decreases exponentially with its frequency rank 123.  
#       In addition longer documents have disproportionately larger vocabularies (Heap's law).
#       Because both are non-linear, the use of absolute term counts, which fail to
#       take collection size into account, can return misleading "fingerprints".
#       Weighting the terms by taking the size of the collection and the document into 
#       account helps us to get around this problem. 
#   
#   3.6.2 How documents are weighted
#       The formula for tf-idf (Term Frequency-Inverse Document Frequency) weight 
#       is composed of TWO terms.  
#       
#       The first, TF, is the NORMALIZED term frequency in A GIVEN DOCUMENT, i.e. "tf 't,d'"
#       where 't' is a given term and 'd' is a given document
#       i.e. the number of time a given term appears in a single given document,
#       divided by the LENGTH of the document measured in words i.e. the number 
#       of ALL terms in the document
#       OBS.  TF = 1+log(Keyword Count)/log(Word Count). 
#       Log is used to make it easier to graph wide frequency ranges from e.g. 1 to 10.000
#           N.B. to get the number This might be written as len(total tokens'd')? 
#           Count_vectorizer automatically generates tokens in a sparse matrix,
#           (i.e. zeros are ignored).  So the the number of "hits"
#           the sum of a row is the sum of the total then you have to sum across 
#           a row to get the length of a given document measures in words
#       
#       The second,IDF, inverse document frequency, is a way to measure the IMPORTANCE 
#       of a given term in the COLLECTION.  The idea is that common terms like "of" 
#       are LESS interesting than rare terms like "circumcision"
#       Taking the INVERSE is a way to give more importance to rare and less to common
#       So the second term is N/df't' (i.e. number of documents in a collection)/number of documents 
#       WITH THE TERM in the collection). An example:  "of" vs. "circumcision".  
#       There are 10 documents, all have "of" N/df'of' = 10/10 = 1 a small number, so less weight
#       Only one document has "circumcision", so N/df'circumcision' is 10/1 = 10 a big number so more weight
#           N.B. How to get N, number of documents in a collection, i.e. the number of rows in the matrix
#           Use my_matrix.shape within Count_vectorizer to get that (rows, columns)
#           N.B. How to get df't', the number of documents which contain the term
#           Take a slice of just one column (the term) Create a BINARY array in numpy, 
#           which shows ones in row where 't' appears and zeroes where does not
#           Sum the column (can do in numpy) and the sum is df't'
#
from sklearn.feature_extraction.text import TfidfTransformer
## 3. Apply TF-IDF weighting
print('---transform method----')
model_tfidf = TfidfTransformer()
guard_tfidf = model_tfidf.fit_transform(guard_vect)

print(' ')
freqs = guard_tfidf.mean(axis=0).A1
top_idxs2 = (-freqs).argsort()[:10]
print(top_idxs2)
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
weighted_top_words = [inverted_vocabulary[idx] for idx in top_idxs2]
print("Top words: " , weighted_top_words)
# use a dataframe to label the colums
# nwsgrp = random.sample(range(0,len(newsgroups.data)), 10) for new random
# why not tfidf_vectorizer=TfidfVectorizer(use_idf=True) ? 
sub_matrix2 = guard_vect[random_sample, :][:, top_idxs2].todense()
df2=pd.DataFrame(columns=weighted_top_words, index=random_sample, data=sub_matrix2 )

# The new, weighted top ten terms are ['brexit', 'people', 'would', 'one', 'party', 
#   'labour', 'trump', 'johnson', 'women', 'may']. They are more interesting and significant than 
#   the unweighted top terms ['people',  'would',  'one',  'brexit',  'new',  'even',
#  'like',  'may',  'party','time'].  One can see that the Opinion section of the
#   Guardian, in the first 10 months of 2019, has been extremely focussed on Brexit,
#   on political parties, especially Labour, on women and on the politicians Trump, 
#   Johnson and May.  Compared to the old top-10, rankings in the new top-10 have 
#   changed and named politicians appear.
top_freqs = []
for f in top_idxs2 :
    top_freqs.append(freqs[f])
df3=pd.DataFrame()
df3['Word']=weighted_top_words
df3['frequency']=top_freqs
print (df3)


### 4. Model and visualize the topics in your subset using Latent Dirichlet Allocation (LDA)
#        on your pre-processed data.
#        Note: Set n_components=4 (nr. of topics) and random_state=0 (so we get 
#        reproducible results on each run).
##  4.1  Import LatentDirichletAllocation
from sklearn.decomposition import LatentDirichletAllocation
model_lda = LatentDirichletAllocation(n_components=4, random_state=0)
guard_lda = model_lda.fit_transform(guard_vect)

##  4.2 Describe the shape of the resulting matrix.
import numpy as np

print('The shape of the LDA matrix is %i rows and %i columns', np.shape(guard_lda)) 
#   N.B.  x documents (rows) x 4 topics (columns)

#   Each document is considered to have a small set of topics and each topic frequently uses only
#   a small set of terms.  The distribution of terms in a given topic is the topic's
#   distinctive fingerprint or signature.  Terms may be found in more than one topic
#   but the overall signature is not the same.
##  4.3 Create a matrix with headings for rows and columns
top_freqs = []
for f in top_idxs2 :
    top_freqs.append(freqs[f])
df3=pd.DataFrame()
df3['Word']=weighted_top_words
df3['frequency']=top_freqs
print (df3)
print(' ') 
