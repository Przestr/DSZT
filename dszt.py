import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re, string
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import nltk
import string
import spacy
nlp = spacy.load('en_core_web_sm')

random_state = 5041

def remove_url(text):
    text = re.sub(r'((?:https?|ftp|file)://[-\w\d+=&@#/%?~|!:;\.,]*)', '', text)
    return text

def remove_HTML(text):
    text = re.sub(r'<.*?>', '', text)
    return text

def remove_num(text):
    text = re.sub(r'\w*\d+\w*', '', text)
    return text

def remove_references(text):
    text = re.sub(r'&[a-zA-Z]+;?', '', text)
    return text

def remove_non_printable(text):
    text = ''.join([word for word in text if word in string.printable])
    return text


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.drop(columns=['id','keyword','location'],inplace=True)

print(train_df["text"].values[1])


print(train_df.shape)


#Wizualizacja różnic różnic w cechach
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# create a new feature for the number of sentences in each Tweet
train_df['liczba_zdań'] = train_df['text'].apply(nltk.tokenize.sent_tokenize).apply(len)
test_df['liczba_zdań'] = test_df['text'].apply(nltk.tokenize.sent_tokenize).apply(len)

# create a new feature for the number of words
train_df['liczba_słów'] = train_df['text'].apply(nltk.tokenize.word_tokenize).apply(len)
test_df['liczba_słów'] = test_df['text'].apply(nltk.tokenize.word_tokenize).apply(len)

# create a new feature for the number of characters excluding white spaces
train_df['liczba_znakow'] = train_df['text'].apply(lambda x: len(x) - x.count(" "))
test_df['liczba_znakow'] = test_df['text'].apply(lambda x: len(x) - x.count(" "))

# define a function that returns the number of hashtags in a string
def hash_count(string):
    words = string.split()
    hashtags = [w for w in words if w.startswith('#')]
    return len(hashtags)
# create a new feature for the number of hashtags
train_df['liczba_hashy'] = train_df['text'].apply(hash_count)
test_df['liczba_hashy'] = test_df['text'].apply(hash_count)

# define a function that returns the number of mentions in a string
def ment_count(string):
    words = string.split()
    mentions = [w for w in words if w.startswith('@')]
    return len(mentions)
# create a new feature for the number of mentions
train_df['liczba_wzmianek'] = train_df['text'].apply(ment_count)
test_df['liczba_wzmianek'] = test_df['text'].apply(ment_count)

# define a function that returns the number of words in all CAPS
def all_caps_count(string):
    words = string.split()
    pattern = re.compile(r'\b[A-Z]+[A-Z]+\b')
    capsWords = [w for w in words if w in re.findall(pattern, string)]
    return len(capsWords)
# create a new feature for the number of words in all CAPS
train_df['liczba_słów_pisanych_duzymi_literami'] = train_df['text'].apply(all_caps_count)
test_df['liczba_słów_pisanych_duzymi_literami'] = test_df['text'].apply(all_caps_count)

# define a function that returns the average length of words
def avg_word_len(string):
    words = string.split()
    total_len = sum([len(words[i]) for i in range(len(words))])
    avg_len = round(total_len / len(words), 2)
    return avg_len
# create a new feature for the average length of words
train_df['średnia_długość_słowa'] = train_df['text'].apply(avg_word_len)
test_df['średnia_długość_słowa'] = test_df['text'].apply(avg_word_len)

# define a function that returns number of non-proper nouns
def noun_count(text, model=nlp):
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN')
# create a new feature for numbers of non-proper nouns
train_df['liczba_rzeczowników'] = train_df['text'].apply(noun_count)
test_df['liczba_rzeczowników'] = test_df['text'].apply(noun_count)

# define a function that returns the percentage of punctuation
def punc_per(text):
    total_count = len(text) - text.count(" ")
    punc_count = sum([1 for c in text if c in string.punctuation])
    if punc_count != 0 and total_count != 0:
        return round(punc_count / total_count * 100, 2)
    else:
        return 0
# create a new feature for the percentage of punctuation in text
train_df['procent_interpunkcyjnych'] = train_df['text'].apply(punc_per)
test_df['procent_interpunkcyjnych'] = test_df['text'].apply(punc_per)

# define a function that returns number of proper nouns with spaCy
def propn_count(text, model=nlp):
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('PROPN')
# create a new feature for numbers of proper nouns
train_df['propn_count'] = train_df['text'].apply(propn_count)
test_df['propn_count'] = test_df['text'].apply(propn_count)

# liczba_url
train_df['liczba_url'] = train_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
test_df['liczba_url'] = test_df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))


# store the features and their names in variables
features = ['liczba_zdań', 'liczba_słów', 'liczba_znakow', 'liczba_hashy', 'liczba_wzmianek', 'liczba_słów_pisanych_duzymi_literami', 
            'średnia_długość_słowa', 'liczba_rzeczowników', 'liczba_url']
            

# create the figure
fig = plt.figure(figsize=(20, 20))

# adjust the height of the padding between subplots to avoid overlapping
plt.subplots_adjust(hspace=0.3)

# add a centered suptitle to the figure
plt.suptitle("Różnice w cechach, katastrofa vs brak katastrofy", fontsize=20, y=0.91)

# generate the histograms in a for loop
for i, feature in enumerate(features):
    
    # add a new subplot iteratively
    ax = plt.subplot(4, 3, i+1)
    ax = train_df[train_df['target']==0][feature].hist(alpha=0.5, label='Brak katastrofy', bins=40, color='royalblue', density=True)
    ax = train_df[train_df['target']==1][feature].hist(alpha=0.5, label='Katastrofa', bins=40, color='lightcoral', density=True)
    
    # set x_label, y_label, and legend
    ax.set_xlabel(features[i], fontsize=14)
    ax.set_ylabel('Gęstość prawd.', fontsize=14)
    ax.legend(loc='upper right', fontsize=14)
    

# shot the figure
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

train_df['text_cleaned'] = train_df['text'].apply(remove_url)
test_df['text_cleaned'] = test_df['text'].apply(remove_url)

train_df['text_cleaned'] = train_df['text_cleaned'].apply(remove_HTML)
test_df['text_cleaned'] = test_df['text_cleaned'].apply(remove_HTML)

train_df['text_cleaned'] = train_df['text_cleaned'].apply(remove_references)
test_df['text_cleaned'] = test_df['text_cleaned'].apply(remove_references)

train_df['text_cleaned'] = train_df['text_cleaned'].apply(remove_non_printable)
test_df['text_cleaned'] = test_df['text_cleaned'].apply(remove_non_printable)

train_df['text_cleaned'] = train_df['text_cleaned'].apply(remove_num)
test_df['text_cleaned'] = test_df['text_cleaned'].apply(remove_num)

train_df['text_cleaned'] = [t.lower() for t in train_df['text_cleaned']]
test_df['text_cleaned'] = [t.lower() for t in test_df['text_cleaned']]


#print(train_df.shape)


count_vectorizer = feature_extraction.text.CountVectorizer()

X=train_df['text_cleaned']
y=train_df['target']

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1, random_state=random_state)

X_train=count_vectorizer.fit_transform(X_train).toarray()
X_test=count_vectorizer.transform(X_test)

clf = linear_model.LogisticRegression(max_iter=10000, random_state=random_state)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

train_accuracy = clf.score(X_train,y_train)
val_accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

print("train accuracy: ", train_accuracy)
print("validation accuracy: ", val_accuracy)
print("f1 score: ", f1score)

#print(classification_report(y_test.values,y_pred))