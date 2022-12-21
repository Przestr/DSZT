import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re, string
from sklearn.metrics import accuracy_score, f1_score

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