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
import string
import osimport joblib
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
        joblib.dump(self.vectorizer, "./models/vectorizer_%s.sav"%self.model_name)
        joblib.dump(self.model,"./models/%s.sav"%self.model_name)
 
    def load(self):
        self.vectorizer = joblib.load("./models/vectorizer_%s.sav"%self.model_name)
        self.model= joblib.load("./models/%s.sav"%self.model_name)
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
    positive_file =open('../../data/sexually_explicit_comments.csv', 'r', encoding='utf-8')
    negative_file =pd.read_csv('../../data/FinalBalancedDataset.csv')

    negatives = [i[-1] for i in negative_file.values if i[1] == 0]
    positives = positive_file.read().splitlines()
    positive_file.close()
    return negatives, positives

def get_racism_data():
    file =pd.read_csv("../../data/cyberbullying_tweets.csv")
    negatives= [row[0] for row in file.values if row[-1] == 'not_cyberbullying']
    positives =[row[0] for row in file.values if row[-1] == 'ethnicity'] 
    return negatives, positives

def get_cyberbullying_data():
    file = pd.read_csv('../../data/cyberbullying_tweets.csv')
    negatives=  [row[0] for row in file.values if row[1] == 'not_cyberbullying']
    positives=  [row[0] for row in file.values if row[1] != 'not_cyberbullying']
    return negatives, positives

def get_sexism_data():
    negative_file = pd.read_csv('../../data/cyberbullying_tweets.csv')
    positive_file = pd.read_csv('../../data/sexist/sexism_data.csv')
    positives =[row[2] for row in positive_file.values if row[4]]       
    negatives= [row[0] for row in negative_file.values if row[-1] == 'not_cyberbullying'] 
    return negatives, positives
def test_and_dump():    os.mkdir('models')
    sexual_content_filters = [BayesianSexualContentFilter(), SexualContentFilter()]
    for filter in sexual_content_filters:
        filter.train()
        filter.dump()
        print(filter("Show the tits", "The weather is so nice today"))
        print(filter.accuracy)

    racism_filters = [BayesianRacismFilter(), RacismFilter()]
    for racism_filter in racism_filters:
        racism_filter.train()
        racism_filter.dump()
        print(racism_filter("Jews and gypsies are the worst, blacks are bit better", "The weather is so nice today"))
        print(racism_filter.accuracy)

    sexism_filters= [BayesianSexismFilter(), SexismFilter()]
    for sexism_filter in sexism_filters:
        sexism_filter.train()
        sexism_filter.dump()
        print(sexism_filter("Place of the woman is kitchen, men are superior to women", "The weather is so nice today"))
        print(sexism_filter.accuracy)

    cbullying_filters = [CyberBullyingFilter(),BayesianCyberBullyingFilter(), ]
    for cyberbullying_filter in cbullying_filters:
        cyberbullying_filter.train()
        cyberbullying_filter.dump()
        print(cyberbullying_filter("You look ugly, go kill yourself", "The weather is so nice today"))
        print(cyberbullying_filter.accuracy)test_and_dump()