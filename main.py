import itertools
from pprint import pprint
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

from preprocessing.preprocessor import Preprocessor


def compare_preprocessing():
    # Loading train and test data:

    all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                      'comp.sys.mac.hardware', 'comp.windows.x',
                      'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
                      'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
                      'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    print("Loading 20 newsgroups...")

    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                          categories=all_categories)

    newsgroups_test = fetch_20newsgroups(subset='test',
                                         remove=('headers', 'footers', 'quotes'),
                                         categories=all_categories)

    print("{} training documents loaded.".format(newsgroups_train.filenames.shape[0]))

    print("Buidling Preprocessor combinations...")
    # flags: special_character_removal, number_removal, url_email_removal, stopword_removal, lower, stemming, lemmatize
    num_of_preprocessor_flags = 7
    # Creates a list of all possible permutations of a boolean list with the length of number of flags
    booleans = [False, True]  # Creates a list of all possible permutations of a boolean list
    flags_list = [list(b) for b in itertools.product(booleans, repeat=num_of_preprocessor_flags)]

    invalid_flags = []
    for i in range(len(flags_list)):
        if flags_list[i][5] and flags_list[i][6]:  # Removes simultaneous Stemming and Lemmatization
            invalid_flags.append(flags_list[i])
        elif flags_list[i][5] and not flags_list[i][4]:  # Remove Stemming without lowercase (lowercase is inbuilt)
            invalid_flags.append(flags_list[i])

    flags_list = [x for x in flags_list if x not in invalid_flags]
    print("{} Combinations built.".format(len(flags_list)))

    # Initialize vectorizer, machine learning algorithm and data frame to store the results
    vectorizer = TfidfVectorizer(analyzer="word", tokenizer=dummy, lowercase=False, preprocessor=dummy, stop_words=None)
    clf = MultinomialNB(alpha=.01)
    columns = ['Special Character Removal', 'Number Removal', 'URL and E-Mail Removal', 'Stopword Removal', 'Lowercase',
               'Stemming', 'Lemmatization', 'Unique Words', 'Accuracy']
    rows = []

    for flags in flags_list:  # loops through all combinations
        prep = Preprocessor(special_character_removal=flags[0], number_removal=flags[1], url_email_removal=flags[2],
                            stopword_removal=flags[3], lower=flags[4], stemming=flags[5], lemmatize=flags[6])

        preprocessed_train_data = [prep.preprocess(d) for d in newsgroups_train.data]

        preprocessed_test_data = [prep.preprocess(d) for d in newsgroups_test.data]

        vectors = vectorizer.fit_transform(preprocessed_train_data)

        # Train machine learning model
        clf.fit(vectors, newsgroups_train.target)

        # Transform test data to the model fitted to the training data
        vectors_test = vectorizer.transform(preprocessed_test_data)

        # Evaluate
        pred = clf.predict(vectors_test)
        vocab = vectors.shape[1]
        accuracy = metrics.accuracy_score(newsgroups_test.target, pred)
        rows.append([flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], vocab, accuracy])

        print(
            "Spec: {} , Numbers: {} , EmailUrl: {} , SWR: {}, low: {}, Stem: {} , Lem: {} -> Vocab: {}, Acc: {}".format(
                flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6], vocab, accuracy))

    # Organize data frame and save the results
    df = pd.DataFrame(np.array(rows), columns=columns)
    df = df.sort_values(by=['Accuracy'], ascending=False)
    pprint(df)
    df.to_csv('results.csv', sep=';')


def dummy(doc):
    return doc


if __name__ == "__main__":
    compare_preprocessing()

