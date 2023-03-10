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
import seaborn as sns
nlp = spacy.load('en_core_web_sm')

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
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

# create a function that builds and  runs multiple classification models
def classify(clf, x_train,x_val, y_train, y_val):
    y_pred = clf.fit(x_train, y_train).predict(x_val)
    train_accuracy = clf.score(x_train,y_train)
    val_accuracy = accuracy_score(y_val, y_pred)
    f1score = f1_score(y_val, y_pred)
    return train_accuracy, val_accuracy, f1score


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df["text"].values[1])


print(train_df.shape)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()

train_df.groupby('target').count()['id'].plot(kind='pie', ax=axes[0], labels=['Brak katastrofy (57%)', 'Katastrofa (43%)'])
sns.countplot(x=train_df['target'], hue=train_df['target'], ax=axes[1])

axes[0].set_ylabel('')
axes[1].set_ylabel('')
axes[1].set_xticklabels(['Brak Katastrofy (4342)', 'Katastrofa (3271)'])
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Target Distribution in Training Set', fontsize=13)
axes[1].set_title('Rozk??ad danych w zbiorze treningowym', fontsize=13)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=train_df[missing_cols].isnull().sum().index, y=train_df[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=test_df[missing_cols].isnull().sum().index, y=test_df[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Liczba tweet??w z brakiem warto??ci', size=15, labelpad=20)
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Zbi??r treningowy', fontsize=13)
axes[1].set_title('Zbi??r testowy', fontsize=13)

plt.show()

for df in [train_df, test_df]:
    for col in ['keyword', 'location']:
        df[col] = df[col].fillna(f'no_{col}')



train_df.drop(columns=['id','keyword','location'],inplace=True)

#Wizualizacja r????nic r????nic w cechach
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# create a new feature for the number of sentences in each Tweet
train_df['liczba_zda??'] = train_df['text'].apply(nltk.tokenize.sent_tokenize).apply(len)
test_df['liczba_zda??'] = test_df['text'].apply(nltk.tokenize.sent_tokenize).apply(len)

# create a new feature for the number of words
train_df['liczba_s????w'] = train_df['text'].apply(nltk.tokenize.word_tokenize).apply(len)
test_df['liczba_s????w'] = test_df['text'].apply(nltk.tokenize.word_tokenize).apply(len)

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
train_df['liczba_s????w_pisanych_duzymi_literami'] = train_df['text'].apply(all_caps_count)
test_df['liczba_s????w_pisanych_duzymi_literami'] = test_df['text'].apply(all_caps_count)

# define a function that returns the average length of words
def avg_word_len(string):
    words = string.split()
    total_len = sum([len(words[i]) for i in range(len(words))])
    avg_len = round(total_len / len(words), 2)
    return avg_len
# create a new feature for the average length of words
train_df['??rednia_d??ugo????_s??owa'] = train_df['text'].apply(avg_word_len)
test_df['??rednia_d??ugo????_s??owa'] = test_df['text'].apply(avg_word_len)

# define a function that returns number of non-proper nouns
def noun_count(text, model=nlp):
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return pos.count('NOUN')
# create a new feature for numbers of non-proper nouns
train_df['liczba_rzeczownik??w'] = train_df['text'].apply(noun_count)
test_df['liczba_rzeczownik??w'] = test_df['text'].apply(noun_count)

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
features = ['liczba_zda??', 'liczba_s????w', 'liczba_znakow', 'liczba_hashy', 'liczba_wzmianek', 'liczba_s????w_pisanych_duzymi_literami', 
            '??rednia_d??ugo????_s??owa', 'liczba_rzeczownik??w', 'liczba_url', 'procent_interpunkcyjnych']
            

# create the figure
fig = plt.figure(figsize=(20, 20))

# adjust the height of the padding between subplots to avoid overlapping
plt.subplots_adjust(hspace=0.3)

# add a centered suptitle to the figure
plt.suptitle("R????nice w cechach, katastrofa vs brak katastrofy", fontsize=20, y=0.91)

# generate the histograms in a for loop
for i, feature in enumerate(features):
    
    # add a new subplot iteratively
    ax = plt.subplot(4, 3, i+1)
    ax = train_df[train_df['target']==0][feature].hist(alpha=0.5, label='Brak katastrofy', bins=40, color='royalblue', density=True)
    ax = train_df[train_df['target']==1][feature].hist(alpha=0.5, label='Katastrofa', bins=40, color='lightcoral', density=True)
    
    # set x_label, y_label, and legend
    ax.set_xlabel(features[i], fontsize=14)
    ax.set_ylabel('G??sto???? prawd.', fontsize=14)
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

#X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.1, random_state=random_state)

t_N_FOLDS = 6
t_X_folds = np.array_split(X, t_N_FOLDS)
t_y_folds = np.array_split(y, t_N_FOLDS)


# create a dictionary containing the names and code of algorithms
clfs = {'MNB': MultinomialNB(),
    #    'SVC': SVC(kernel='sigmoid', gamma=1.0, random_state=random_state),
        'LR': LogisticRegression(max_iter=10000, random_state=random_state),
        'KNC': KNeighborsClassifier(),
       'RFC': RandomForestClassifier(random_state=random_state),
       'ETC': ExtraTreesClassifier(random_state=random_state),
       'BC': BaggingClassifier(random_state=random_state)}
    #    'XGBC': XGBClassifier(n_estimators=50, random_state=random_state)}

# create lists to store scores to build a dataframe later on
val_accuracy_series = []
f1score_series = []

val_accuracy_series_mean = []
f1score_series_mean = []

# run the models with classify() function we created above(this takes some times)
for name, clf in clfs.items():
    print('For [{}]-\n'.format(name))
    for x in range(t_N_FOLDS):
        t_X_train = t_X_folds.copy()
        t_X_train.pop(x)
        t_X_train = np.concatenate(t_X_train)
        t_y_train = t_y_folds.copy()
        t_y_train.pop(x)
        t_y_train = np.concatenate(t_y_train)

        t_X_test = t_X_folds[x]
        t_y_test = t_y_folds[x]

        X_train=count_vectorizer.fit_transform(t_X_train).toarray()
        X_test=count_vectorizer.transform(t_X_test)

        i_train_accuracy, i_val_accuracy, i_f1score = classify(clf, X_train, X_test, t_y_train, t_y_test)

        # append the scores of each fold
        val_accuracy_series.append(i_val_accuracy)
        f1score_series.append(i_f1score)
        
        # print out the scores
        print('Fold : {} | Train accuracy : {} | Val accuracy : {} | F1 Score : {}\n'.format(x+1,
                                                                                    round(i_train_accuracy,2),
                                                                                    round(i_val_accuracy,2),
                                                                                    round(i_f1score,2)))
        if (x + 1) == t_N_FOLDS:
            val_accuracy_series_mean.append(np.mean(val_accuracy_series))
            f1score_series_mean.append(np.mean(f1score_series))
            val_accuracy_series = []
            f1score_series = []

# create a dataframe with the scores
performance_df = pd.DataFrame({'Algorithm': clfs.keys(),
                         'Accuracy': val_accuracy_series_mean,
                         'F1score': f1score_series_mean}).round(2).sort_values('F1score', ascending=False)

print(performance_df)

# clf = linear_model.LogisticRegression(max_iter=10000, random_state=random_state)
# clf.fit(X_train, y_train)
# y_pred=clf.predict(X_test)

# train_accuracy = clf.score(X_train,y_train)
# val_accuracy = accuracy_score(y_test, y_pred)
# f1score = f1_score(y_test, y_pred)

# print("train accuracy: ", train_accuracy)
# print("validation accuracy: ", val_accuracy)
# print("f1 score: ", f1score)

#print(classification_report(y_test.values,y_pred))