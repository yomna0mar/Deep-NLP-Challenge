import pandas as pd
import nltk

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
from sklearn.svm import LinearSVC
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        """
        Constructor that takes several classifiers as input
        Goal: using all the input classifiers, classify the object as "flagged" or "not_flagged" by taking the most agreed upon class
        """
        self.classifiers = classifiers;
        
    def classify(self, sample):
        """
        Inherits the classify function from ClassifierI
        """
        votes = [];
        for c in self.classifiers:
            v = c.classify(sample);
            votes.append(v);    
        return mode(votes);
    
    def agreement_ratio(self, sample):
        """
        Returns the agreement ratio of a certain classification
        For example, if all classifiers agree that a sample is "not-flagged", then the agreement ratio is 1
        The agreement ratio is a number between 0 and 1
        """
        votes = [];
        for c in self.classifiers:
            v = c.classify(sample);
            votes.append(v);
        return (votes.count(mode(votes)))/(len(votes));
    
def extract_data():
    """
    Extracts both datasets from their destination folder
    Output: both datasets as lists, where the text is tokenized by words and has its stopwords removed
    """
    chatbot_dataset = pd.read_csv("Datasets/chatbot.csv", usecols=["class","response_text"], encoding='latin-1');
    resumes_dataset = pd.read_csv("Datasets/resumes.csv", usecols=["class","resume_text"], encoding='latin-1');
    
    chatbot_dataset = chatbot_dataset.values.tolist();
    resumes_dataset = resumes_dataset.values.tolist();
    
    stop_words = set(nltk.corpus.stopwords.words("english"));
    
    for i in chatbot_dataset:
        i[1] = nltk.word_tokenize(i[1]);
        i[1] = [w for w in i[1] if not w in stop_words];
        
    for i in resumes_dataset:
        i[1] = nltk.word_tokenize(i[1]);
        i[1] = [w for w in i[1] if not w in stop_words];
        
    return chatbot_dataset, resumes_dataset;

def train_test_split(dataset, train_ratio):
    """
    Splits the given dataset into training and testing sets
    Input: dataset to be split and the split ratio of the training set
    Output: the train and test sets for the given dataset
    """
    train = round(train_ratio * len(dataset));    
    return dataset[:train], dataset[train:];
    
def get_most_common_words(dataset, num):
    """
    Returns the most common words in the given dataset
    Input: a dataset and num is how many most common words needed
    Output: a list of the top "num" most common words in the dataset in order
    """
    all_words = [];
    for i in dataset:
        for j in i[1]:
            all_words.append(j.lower());
    all_words = nltk.FreqDist(all_words);
    
    most_common_words = [];
    for i in all_words.most_common(num):
        most_common_words.append(i[0]);
        
    return most_common_words;

def extract_features(comment, most_common_words):
    """
    Determine whether the input comment contains some of the most_common_words
    Input: a comment or set of words and/or sentences and a list of the most common words in the original dataset
    Output: a dictonary, where the keys are the most common words and the values are True/False
            True when the input comment contains the words, False otherwise
    """
    words = set(comment);
    features = {};
    
    for w in most_common_words:
        features[w] = (w in words);
        
    return features;

def data_preprocessing(dataset, most_common_words):
    """
    Converts the "comment" or "words" of the dataset's instances to a dictionary that determines if the comment has any words in the most_common_words
    Input: a dataset and its top 250 most common words
    Output: the same dataset restructured in a more appealing way in order to perform data analysis
    """
    processed_dataset = []
    for i in dataset:
        temp = extract_features(i[1], most_common_words);
        processed_dataset.append((temp, i[0]));
    
    return processed_dataset;

if __name__ == "__main__":
    """
    Main Program
    """
    chatbot, resumes = extract_data();
    
    chatbot_most_common_words = get_most_common_words(chatbot, 250);
    resumes_most_common_words = get_most_common_words(resumes, 500);
    
    chatbot_dataset = data_preprocessing(chatbot, chatbot_most_common_words);
    resumes_dataset = data_preprocessing(resumes, resumes_most_common_words);
    
    chatbot_train, chatbot_test = train_test_split(chatbot_dataset, 0.8);
    resumes_train, resumes_test = train_test_split(resumes_dataset, 0.8);
    
    # Naive Bayes
    chatbot_nb = nltk.NaiveBayesClassifier.train(chatbot_train);
    resumes_nb = nltk.NaiveBayesClassifier.train(resumes_train);
    print("Naive Bayes Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_nb, chatbot_test)*100, "%");
    print("Naive Bayes Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_nb, resumes_test)*100, "%");
    
    # Multinomial Naive Bayes
    mnb = SklearnClassifier(MultinomialNB());
    chatbot_mnb = mnb.train(chatbot_train);
    resumes_mnb = mnb.train(resumes_train);
    print("\nMultinomial Naive Bayes Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_mnb, chatbot_test)*100, "%");
    print("Multinomial Naive Bayes Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_mnb, resumes_test)*100, "%");
    
    # Bernoulli Naive Bayes
    bnb = SklearnClassifier(BernoulliNB());
    chatbot_bnb = bnb.train(chatbot_train);
    resumes_bnb = bnb.train(resumes_train);
    print("\nBernoulli Naive Bayes Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_bnb, chatbot_test)*100, "%");
    print("Bernoulli Naive Bayes Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_bnb, resumes_test)*100, "%");
    
    # Random Forest
    rf = SklearnClassifier(RandomForestClassifier());
    chatbot_rf = rf.train(chatbot_train);
    resumes_rf = rf.train(resumes_train);
    print("\nRandom Forest Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_rf, chatbot_test)*100, "%");
    print("Random Forest Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_rf, resumes_test)*100, "%");
    
    # Logistic Regression
    lr = SklearnClassifier(LogisticRegression());
    chatbot_lr = lr.train(chatbot_train);
    resumes_lr = lr.train(resumes_train);
    print("\nLogistic Regression Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_lr, chatbot_test)*100, "%");
    print("Logistic Regression Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_lr, resumes_test)*100, "%");
    
    # Stochastic Gradient Descent Classifier
    sgd = SklearnClassifier(SGDClassifier());
    chatbot_sgd = sgd.train(chatbot_train);
    resumes_sgd = sgd.train(resumes_train);
    print("\nStochastic Gradient Descent Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_sgd, chatbot_test)*100, "%");
    print("Stochastic Gradient Descent Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_sgd, resumes_test)*100, "%");
    
    # Linear Support Vector Machine
    linear_svm = SklearnClassifier(LinearSVC());
    chatbot_linear_svm = linear_svm.train(chatbot_train);
    resumes_linear_svm = linear_svm.train(resumes_train);
    print("\nLinear SVM Classification Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_linear_svm, chatbot_test)*100, "%");
    print("Linear SVM Classification Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_linear_svm, resumes_test)*100, "%");
    
    # Vote Classifier
    # For the chatbot dataset, we will pick the top 5 most accurate algorithms: Naive Bayes, Multinomial Naive Bayes, Random Forest, Logistic Regression, and Linear SVM
    # For the resumes dataset, we will pick the top 3 most accurate algorithms: Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes
    chatbot_vote_classifier = VoteClassifier(chatbot_nb, chatbot_mnb, chatbot_rf, chatbot_lr, chatbot_linear_svm);
    resumes_vote_classifier = VoteClassifier(resumes_nb, resumes_mnb, resumes_bnb);
    print("\nVoted Classifier Accuracy for the Chatbot Dataset:", nltk.classify.accuracy(chatbot_vote_classifier, chatbot_test)*100, "%");
    print("Voted Classifier Accuracy for the Resumes Dataset:", nltk.classify.accuracy(resumes_vote_classifier, resumes_test)*100, "%\n");