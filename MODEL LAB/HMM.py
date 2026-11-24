import nltk
import numpy as np
from collections import Counter, defaultdict
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics

nltk.download('treebank')

data = nltk.corpus.treebank.tagged_sents()[:100]
train, test = data[:80], data[80:]

# HMM training
transition = defaultdict(Counter)
emission = defaultdict(Counter)
tag_count = Counter()
tags = set()

for sentence in train:
    previous = "START"
    for word, tag in sentence:
        transition[previous][tag] += 1
        emission[tag][word.lower()] += 1
        tag_count[tag] += 1
        tags.add(tag)
        previous = tag

def viterbi(words):
    dp = [{}]
    back = [{}]

    for tag in tags:
        dp[0][tag] = np.log(
            (transition["START"][tag] + 1) /
            (sum(transition["START"].values()) + len(tags))
        ) + np.log(
            (emission[tag][words[0].lower()] + 1) /
            (tag_count[tag] + 1000)
        )

    for i in range(1, len(words)):
        dp.append({})
        back.append({})
        for tag in tags:
            scores = {
                prev: dp[i-1][prev] +
                np.log((transition[prev][tag] + 1) /
                (sum(transition[prev].values()) + len(tags))) +
                np.log((emission[tag][words[i].lower()] + 1) /
                (tag_count[tag] + 1000))
                for prev in tags
            }
            best = max(scores, key=scores.get)
            dp[i][tag] = scores[best]
            back[i][tag] = best

    last = max(dp[-1], key=dp[-1].get)
    result = [last]

    for i in range(len(words)-1, 0, -1):
        last = back[i][last]
        result.append(last)

    return result[::-1]

# CRF features
def features(sentence, neighbors=True):
    result = []
    for i, (word, tag) in enumerate(sentence):
        w = word.lower()
        f = {
            'word': w,
            'suffix': w[-3:],
            'prefix': w[:2],
            'upper': word.isupper(),
            'digit': word.isdigit()
        }
        if neighbors:
            f['previous'] = sentence[i-1][0].lower() if i > 0 else 'START'
            f['next'] = sentence[i+1][0].lower() if i < len(sentence)-1 else 'END'
        result.append(f)
    return result

X_train = [features(s) for s in train]
X_test = [features(s) for s in test]
y_train = [[t for w, t in s] for s in train]
y_test = [[t for w, t in s] for s in test]

# CRF with neighbouring features
crf = CRF(max_iterations=50, algorithm='lbfgs')
crf.fit(X_train, y_train)
pred = crf.predict(X_test)

print("CRF Accuracy:", metrics.flat_accuracy_score(y_test, pred))
print("Accuracy per tag:")
print(metrics.flat_classification_report(y_test, pred))

# CRF without neighbouring features
X_train2 = [features(s, False) for s in train]
X_test2 = [features(s, False) for s in test]

crf2 = CRF(max_iterations=50, algorithm='lbfgs')
crf2.fit(X_train2, y_train)
pred2 = crf2.predict(X_test2)

print("CRF without neighbours:",
      metrics.flat_accuracy_score(y_test, pred2))

# HMM known and unseen word
words = ["The", "market", "rose"]
print("HMM known words:", viterbi(words))

unknown = ["The", "xyzword", "rose"]
print("HMM unseen word:", viterbi(unknown))