from preprocessing import *
import fasttext
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import spacy
# from nltk import word_tokenize

# nlp = spacy.load("en_core_web_md")
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

# Get the data
df_train, df_test = get_data()
	
# Remove empty sentences after preprocessing
df_train = df_train[df_train.processed_tweet!=""]


# To be used to check validation accuracy
# traindf, valdf = train_test_split(df_train,test_size=0.2,train_size=0.8)
# traindf.reset_index(inplace=True,drop=True)
# valdf.reset_index(inplace=True,drop=True)


# TFIDF vectors with Random Forrest Classifier
vectorizer = TfidfVectorizer(max_df = 0.8,min_df = 0.05)
x_train = vectorizer.fit_transform(df_train.processed_tweet)
x_test = vectorizer.transform(df_test.processed_tweet).todense().tolist()
y_train = df_train.hateful


# Train Random Forest Model
rfc = RandomForestClassifier(class_weight='balanced')
rfc.fit(x_train, y_train)


# pred = rfc.predict(x_test)
df1 = df_test.copy()
df1["hateful"] = rfc.predict(x_test)
df1.drop(["processed_tweet"], axis=1, inplace=True)
df1.to_csv("Random Forest.csv", index=False)
# print(accuracy_score(valdf.hateful,pred))


# Uncomment to check lastest string length
# l = [len(word_tokenize(sentence)) for sentence in df_train.processed_tweet]
# sorted(l,reverse=True)[:10]


if os.path.isfile("x_trainscapy.pkl"):
    with open("x_trainscapy.pkl","rb") as f:
        x_train = pickle.load(f)
else:
    x_train = [sum([token.vector for token in nlp(sentence)])/len(nlp(sentence)) for sentence in df_train.processed_tweet]
if os.path.isfile("x_testscapy.pkl"):
    with open("x_testscapy.pkl","rb") as f:
        x_test = pickle.load(f)
else:
    x_test = [sum([token.vector for token in nlp(sentence)])/len(nlp(sentence)) for sentence in df_test.processed_tweet]

y_train = df_train.hateful


# with open("x_trainscapy.pkl","wb") as f:
#     pickle.dump(x_train,f)

# with open("x_testscapy.pkl","wb") as f:
#     pickle.dump(x_test,f)


clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)


# pred = clf.predict(x_test)
df1 = df_test.copy()
df1["hateful"] = clf.predict(x_test)
df1.drop(["processed_tweet"], axis=1, inplace=True)
df1.to_csv("Support Vector Machine.csv", index=False)


# Fasttext model
# Create the data file
with open("data_train.txt","w") as f:
    for _,row in df_train.iterrows():
        f.write(row["processed_tweet"] + " __label__"+ str(row.hateful)+ "\n")

# Train the model
model = fasttext.train_supervised('data_train.txt')

# Predict
predlabel = np.array(model.predict(df_test.processed_tweet.to_list())[0])
test_labels = [int(l[0][-1]) for l in predlabel]

# Write labels
df1 = df_test.copy()
df1["hateful"] = test_labels
df1.drop(["processed_tweet"], axis=1, inplace=True)
df1.to_csv("Fasttext.csv", index=False)

