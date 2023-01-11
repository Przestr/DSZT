import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re, string
from sklearn.metrics import accuracy_score, f1_score

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
train_df.drop(columns=['id','keyword','location'],inplace=True)

print(train_df["text"].values[1])


print(train_df.shape)

#print(test_df.tail())

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

# run the models with classify() function we created above(this takes some times)
for name, clf in clfs.items():
    i_train_accuracy, i_val_accuracy, i_f1score = classify(clf, X_train, X_test, y_train, y_test)
    
    # append the scores to the lists
    val_accuracy_series.append(i_val_accuracy)
    f1score_series.append(i_f1score)
    
    # print out the scores
    print('For [{}]-\nTrain accuracy : {} | Val accuracy : {} | F1 Score : {}\n'.format(name,
                                                                                  round(i_train_accuracy,2),
                                                                                  round(i_val_accuracy,2),
                                                                                  round(i_f1score,2)))

# create a dataframe with the scores
performance_df = pd.DataFrame({'Algorithm': clfs.keys(),
                         'Accuracy': val_accuracy_series,
                         'F1score': f1score_series}).round(2).sort_values('F1score', ascending=False)

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