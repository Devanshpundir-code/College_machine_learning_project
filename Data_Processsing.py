import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("darkpattern.csv")   # update path

# Drop missing values
df = df.dropna()

# Separate features and labels
X = df['text']      
y = df['label']

# Encode labels
le = LabelEncoder()
y_binary = le.fit_transform(y)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)
