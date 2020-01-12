import numpy as np
import pandas as pd
import nltk
import sklearn
import operator
#from joblib import dump, load  # for saving the SVM to disk
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

# Requirements for functions:
stopwords = set(nltk.corpus.stopwords.words('english'))
extra_stopwords = [".", ",", "--", "``", "’", "&", "!", "?", "@", ":", ";", "'", "-", "https", "http", "#", "..."]
for stopword in extra_stopwords:
    stopwords.add(stopword)

lemmatizer = nltk.stem.WordNetLemmatizer()


# Functions:

def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


def get_vector_text(list_vocab, string):
    vector_text = np.zeros(len(list_vocab))
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    return vector_text


def get_vocabulary(dataset, n_features):
    dict_word_frequency = {}
    for tweet in dataset:
        sentence_tokens = get_list_tokens(tweet[0])
        for word in sentence_tokens:
            if word in stopwords:
                continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1

    # Now we create a sorted frequency list with the top 1000 words, using the function "sorted". Let's see the 15 most frequent words
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:n_features]
    # Finally, we create our vocabulary based on the sorted frequency list
    vocabulary = []
    for word, frequency in sorted_list:
        vocabulary.append(word)

    return vocabulary


def train_svm_classifier(training_set, vocabulary):  # Function for training our svm classifier
    X_train = []
    Y_train = []
    for instance in training_set:
        vector_instance = get_vector_text(vocabulary, instance[0])
        X_train.append(vector_instance)
        Y_train.append(instance[1])
    # Finally, we train the SVM classifier
    svm_clf = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf.fit(np.asarray(X_train), np.asarray(Y_train))
    return svm_clf


print("Loading data...")
tweet_data = pd.read_csv("./Hateval/hateval.tsv", delimiter="\t", comment=None, encoding="utf8")
all_tweets = tweet_data["text"].tolist()
all_labels = tweet_data["label"].tolist()
tweets_labels_all = [list(tup) for tup in zip(all_tweets, all_labels)]  # combine into tuples

print("Substituting some text speech and HTML encoding for true English and unicode...")
sub_words = [(" u ", " you "), (" ur ", " your"), (" r ", " are "), (" &amp; ", " & ")]

for word in sub_words:
    for i, tweet in enumerate(tweets_labels_all):
        tweets_labels_all[i][0] = tweet[0].replace(word[0], word[1])
print("Done")

while True:
    try:
        features = int(input("How many features to use? [500]:") or "500")
        k_folds = int(input("How many folds? [10]:") or "10")
    except ValueError:
        "Not a number!"
    else:
        break


print("Performing {}-fold cross-validation...".format(k_folds))

results = pd.DataFrame(columns=["fold", "accuracy", "precision", "recall", "F1_score"])


folds_current = 0
kf = KFold(n_splits=k_folds)
random.shuffle(tweets_labels_all)
kf.get_n_splits(tweets_labels_all)
for train_index, test_index in kf.split(tweets_labels_all):
    train_set_fold = []
    test_set_fold = []
    accuracy_total = 0.0
    for i, instance in enumerate(tweets_labels_all):
        if i in train_index:
            train_set_fold.append(instance)
        else:
            test_set_fold.append(instance)
    vocabulary_fold = get_vocabulary(train_set_fold, features)
    svm_clf_fold = train_svm_classifier(train_set_fold, vocabulary_fold)
    X_test_fold = []
    Y_test_fold = []
    for instance in test_set_fold:
        vector_instance = get_vector_text(vocabulary_fold, instance[0])
        X_test_fold.append(vector_instance)
        Y_test_fold.append(instance[1])
    Y_test_fold_gold = np.asarray(Y_test_fold)
    X_test_fold = np.asarray(X_test_fold)
    Y_test_predictions_fold = svm_clf_fold.predict(X_test_fold)
    accuracy_fold = accuracy_score(Y_test_fold_gold, Y_test_predictions_fold)
    precision_fold = precision_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
    recall_fold = recall_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
    F1_score_fold = f1_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
    results.loc[folds_current] = ["fold " + str(folds_current),
                                  accuracy_fold,
                                  precision_fold,
                                  recall_fold,
                                  F1_score_fold
                                  ]
    folds_current += 1
    print("Fold {} of {} completed.".format(folds_current, k_folds))

results.loc[folds_current] = ["average",
                              np.mean(results["accuracy"]),
                              np.mean(results["precision"]),
                              np.mean(results["recall"]),
                              np.mean(results["F1_score"])
                              ]

print(results)
