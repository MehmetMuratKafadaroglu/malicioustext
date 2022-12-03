import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import string

_stopwords = set(stopwords.words('english'))
_lemmatizer =  WordNetLemmatizer()

def tokenize(doc:str):
    return [_lemmatizer.lemmatize(tkn) for tkn in \
            word_tokenize(doc.translate(str.maketrans(' ', ' ', string.punctuation)).lower())\
            if not (tkn in _stopwords or is_link_or_at(tkn))] 

def is_link_or_at(token:str):
    return token[0] == '@' or token[:7] == 'https:/'


VALIDATION_SPLIT = .1
class Filter:
    error_message= "This method must be overwritten"
    def __init__(self,model_name) -> None:
        self.model_name= model_name
        self.accuracy = None 

    @property
    def vectorizer(self):
        raise NotImplementedError(Filter.error_message)
    @vectorizer.setter
    def vectorizer(self, val):
        raise NotImplementedError(Filter.error_message)
      
    @property
    def positives(self):
        raise NotImplementedError(Filter.error_message)

    @property
    def negatives(self): 
        raise NotImplementedError(Filter.error_message)

    @property
    def model(self):
        raise NotImplementedError(Filter.error_message)
    
    @vectorizer.setter
    def model(self, val):
        raise NotImplementedError(Filter.error_message)


    def __call__(self, *messages):
        vectors = self.vectorizer.transform(messages)
        prediction = self.model.predict(vectors)
        return dict(zip(messages, map(bool, prediction)))

    def train(self):
        labels = ([1] * len(self.positives)) + [0] * len(self.negatives)
        features = self.vectorizer.fit_transform(self.positives + self.negatives)
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels,test_size=VALIDATION_SPLIT, shuffle= True)
        self.model.fit(features_train, labels_train)
        self.accuracy = accuracy_score(labels_test, self.model.predict(features_test))
        return self

    def dump(self):
        pickle.dump(self.vectorizer, open("../models/vectorizer_%s.sav"%self.model_name, 'wb'))
        pickle.dump(self.model, open("../models/%s.sav"%self.model_name, 'wb'))
 
    def load(self):
        self.vectorizer = pickle.load(open("../models/vectorizer_%s.sav"%self.model_name, 'rb'))
        self.model= pickle.load(open("../models/%s.sav"%self.model_name, 'rb'))
        return self


class Vectorizer:
    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(tokenizer=tokenize)
        self.svd = TruncatedSVD(n_components= 10)    

    def fit_transform(self, docs):
        X = self.tfidf.fit_transform(docs)
        vals = self.svd.fit_transform(X)
        return vals

    def transform(self, docs):
        X = self.tfidf.transform(docs)
        vals = self.svd.transform(X)
        return vals


class BayesianFilter(Filter):
    def __init__(self, model_name) -> None:
        self.__model =  naive_bayes.MultinomialNB()
        self.__vectorizer = TfidfVectorizer(tokenizer=tokenize)
        super().__init__("bayesian_"+model_name)

    @property
    def model(self):
        return self.__model

    @property
    def vectorizer(self):
        return self.__vectorizer

    @vectorizer.setter
    def vectorizer(self, val):
        self.__vectorizer =val

    @model.setter
    def model(self,val):
        self.__model = val


class RandomForestFilter(Filter):
    def __init__(self, model_name) -> None:
        self.__model = RandomForestClassifier(max_depth=100)
        self.__vectorizer = Vectorizer()
        super().__init__("decision_"+model_name,)              

    @property
    def model(self):
        return self.__model

    @property
    def vectorizer(self):
        return self.__vectorizer

    @vectorizer.setter
    def vectorizer(self, val):
        self.__vectorizer =val

    @model.setter
    def model(self,val):
        self.__model = val
 
class BayesianSexualContentFilter(BayesianFilter):
    def __init__(self) -> None:
        super().__init__(model_name="sexual_content_filter")
        self.__negatives, self.__positives = get_sexual_data()

    @property
    def negatives(self):
        return self.__negatives

    @property
    def positives(self):
        return self.__positives

class SexualContentFilter(RandomForestFilter):
    def __init__(self) -> None:
        super().__init__(model_name="sexual_content_filter")
        self.__negatives, self.__positives = get_sexual_data()
    @property
    def negatives(self):
        return self.__negatives

    @property
    def positives(self):
        return self.__positives

class BayesianRacismFilter(BayesianFilter):
    def __init__(self) -> None:
        super().__init__("racism_filter")
        self.__negatives, self.__positives = get_racism_data()
        
    @property
    def negatives(self):
        return self.__negatives 

    @property
    def positives(self):
        return self.__positives 

class RacismFilter(RandomForestFilter):
    def __init__(self) -> None:
        super().__init__("racism_filter")
        self.__negatives, self.__positives = get_racism_data()
        
    @property
    def negatives(self):
        return self.__negatives 

    @property
    def positives(self):
        return self.__positives 


class BayesianSexismFilter(BayesianFilter):
    def __init__(self) -> None:
        super().__init__("sexism_filter")
        self.__negatives, self.__positives= get_sexism_data()
        
    @property
    def positives(self):
        return self.__positives

    @property
    def negatives(self):
        return self.__negatives  

class SexismFilter(RandomForestFilter):
    def __init__(self) -> None:
        super().__init__("sexism_filter")
        self.__negatives, self.__positives= get_sexism_data()
        
    @property
    def positives(self):
        return self.__positives

    @property
    def negatives(self):
        return self.__negatives  
    
class BayesianCyberBullyingFilter(BayesianFilter):
    def __init__(self) -> None:
        super().__init__("cyberbullying_filter")
        self.__negatives, self.__positives= get_cyberbullying_data()

    @property
    def negatives(self):
        return self.__negatives 

    @property
    def positives(self):
        return self.__positives

class CyberBullyingFilter(RandomForestFilter):
    def __init__(self) -> None:
        super().__init__("cyberbullying_filter")
        self.__negatives, self.__positives= get_cyberbullying_data()

    @property
    def negatives(self):
        return self.__negatives 

    @property
    def positives(self):
        return self.__positives


def get_sexual_data():
    positive_file =open('../../../data/sexually_explicit_comments.csv', 'r', encoding='utf-8')
    negative_file =pd.read_csv('../../../data/FinalBalancedDataset.csv')

    negatives = [i[-1] for i in negative_file.values if i[1] == 0]
    positives = positive_file.read().splitlines()
    positive_file.close()
    return negatives, positives

def get_racism_data():
    file =pd.read_csv("../../../data/cyberbullying_tweets.csv")
    negatives= [row[0] for row in file.values if row[-1] == 'not_cyberbullying']
    positives =[row[0] for row in file.values if row[-1] == 'ethnicity'] 
    return negatives, positives

def get_cyberbullying_data():
    file = pd.read_csv('../../../data/cyberbullying_tweets.csv')
    negatives=  [row[0] for row in file.values if row[1] == 'not_cyberbullying']
    positives=  [row[0] for row in file.values if row[1] != 'not_cyberbullying']
    return negatives, positives

def get_sexism_data():
    negative_file = pd.read_csv('../../../data/cyberbullying_tweets.csv')
    positive_file = pd.read_csv('../../../data/sexist/sexism_data.csv')
    positives =[row[2] for row in positive_file.values if row[4]]       
    negatives= [row[0] for row in negative_file.values if row[-1] == 'not_cyberbullying'] 
    return negatives, positives
