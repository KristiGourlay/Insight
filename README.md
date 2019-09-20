# Time Period Prediction Based on NLP

My project will take excerpts from books from various time periods and predict when they were written. The final model will
analyze documents from similar time periods for similarities in both vocabulary and style. The flask app will accept an excerpt from a piece of work
and predict what time period the writing came from.

<img src='projectname/static/images/books.png' alt='beerphoto'>


# Motivation

The usefulness of this model will be its ability to aid in the dating of historical documents. Archives and manuscript collections are known for the bevy of unlabeled documents. Even after OCR and HTR systems, loose documents still need academics with domain expertise to provide dates and then context. Furthermore, on a greater level, this model with provide us with a greater understanding of our language and its evolution over the last hundreds of years.


# Data

I currently have over 30,000 text documents scraped from Project Gutenberg. Due to the sheer size of them, I decided to start with 3000 documents. After
using regex to extract dates, many of the documents still did not have a date. After manually entering dates, and manually entering some data, I
had 1700 text documents. I am still attempting to find more sources to supplement the data that I currently have.

# Process

<img src='projectname/static/images/work_flow.png' alt='work_flow'>

# Targets

After some trial and error, I decided to go with time periods of Literature generally accepted in the Literary field (number in brackets refers to number of entries belonging to that period).

                      Before 1600: Renaissance (24)
                      1670-1800: Enlightenment (73)
                      1830-1870: Romantic Period (305)
                      1870-1920: Victorian (Realism and Naturalism) (934)
                      1910-1945: Modernism (52)
                      1945-present: Contemporary (116)

# Text preparation

First, I had to deal with an imbalance learn. Because a majority of my books come from the 1870-1920 time period, any model would predict that time period for every item. Thus, I had to first balance the target classes by under-sampling and over-sampling. I hope to limit this technique in the following week by finding more sources for the under-represented classes.

Second, I prepared all the data for modeling. This included stemming, lemmatizing, using regex, and lowercasing.

# Modeling

The project is attempting to use many different NLP models to date text documents. So far I have used CountVectorizer and Tf-Idf paired with Logistic Regression, Random Forrest, and Catboost. I have also used Doc2Words. Next step will be to attempt to use BERT and deep learning.
