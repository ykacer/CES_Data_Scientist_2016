from time import time
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# Load some categories from the training set
#categories = ['alt.atheism','comp.graphics']
categories = ['comp.graphics']
#categories = ['alt.atheism']

print "Loading 20 newsgroups dataset for categories:"
print categories

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             data_home='../trash',
                             shuffle=True, random_state=42)

print "%d documents" % len(dataset.data)
print "%d categories" % len(dataset.target_names)

# Print the content of a message
print dataset.data[0]
labels = dataset.target
#true_k = np.unique(labels).shape[0]
true_k=2
print "Extracting features from the training set using a sparse vectorizer"
t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             stop_words='english')
X = vectorizer.fit_transform(dataset.data)
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape


# Do the actual clustering with online K-Means
km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                     init_size=1000, batch_size=500, verbose=1)

print "Clustering sparse data with %s" % km
t0 = time()
km.fit(X)
print "done in %0.3fs" % (time() - t0)

# Look at terms that are the most present in each class
feature_names = vectorizer.get_feature_names()
n_top_words = 10
for k, centroid in enumerate(km.cluster_centers_):
    print "Cluster #%d:" % k
    print " ".join([feature_names[i]
                    for i in centroid.argsort()[:-n_top_words - 1:-1]])

