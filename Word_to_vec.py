import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
df = pd.read_csv("C:/Users/harsh/Downloads/SpamClassifier-master/SpamClassifier-master/smsspamcollection"
                 "/SMSSpamCollection", sep="\t",
                 names=['Label', 'Text'])
print(df.head())
print(len(df["Text"]))
text_data = []
lt = WordNetLemmatizer()
for messages in df["Text"]:
    line = []
    messages = re.sub('[^a-zA-Z]', string=messages, repl=' ')
    message = messages.split(sep=' ')
    for words in message:
        if words not in stopwords.words('english') and words != "":
            words = words.lower()
            words = lt.lemmatize(words)
            line.append(words)
    text_data.append(line)

# WORD2VEC
word_vector_model = Word2Vec(text_data, vector_size=600, min_count=1, window=3, workers=5)
word_vector = word_vector_model.wv
print(word_vector.most_similar("song"))
