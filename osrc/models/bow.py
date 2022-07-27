from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

class BagOfWordsClassifier():

    def __init__(self, parameters:{}):
        self.TYPE = "BagOfWordsClassifier"
        self.clf = LogisticRegression()
        self.vec = CountVectorizer(stop_words='english')
    
    def convert_ds_to_text_array(self, ds:Dataset):
        qca_concats = []
        for (q, c, a, label, evidence) in ds:
            qca_concats.append(' '.join(q + [c] + a))
        return qca_concats
    
    def preprocess(self, ds:Dataset, ts:Dataset()=None):

        # join question, context and answer tokens
        qca_concats = self.convert_ds_to_text_array(ds)
        labels = ds.labels
        
        # include testset in voc!
        if ts:
            qca_concats.extend(self.convert_ds_to_text_array(ts))
            labels.extend(ts.labels)
                
        # count
        x = self.vec.fit_transform(qca_concats)
        return x, labels

    def train(self, ds:Dataset, proba=False):
        data, labels = self.preprocess(ds)
        self.clf.fit(data, labels)
        if proba: outputs, attentions = self.predict_proba(ds)
        else: outputs, attentions = self.predict(ds)

        return {
            'outputs': outputs,
            'attentions': attentions,
            'model': self
            }

    def get_weights(self, ds:Dataset()):
        coefs = self.clf.coef_
        voc = self.vec.vocabulary_
        attentions = []
        # extracting attention
        for (q,c,a,l,e) in ds:
            attn = [0] * len(q)
            for i,token in enumerate(q):
                t = token.lower()
                if t in voc:
                    pos = voc[t]
                    weight = coefs[int(l)][pos]
                    attn[i] = weight

            attentions.append(attn)
        return attentions

    def predict(self, ds:Dataset):
        text_form = self.convert_ds_to_text_array(ds)
        data = self.vec.transform(text_form)
        predictions = self.clf.predict(data)
        return predictions, self.get_weights(ds)
    
    def predict_proba(self, ds:Dataset):
        txt = self.convert_ds_to_text_array(ds)
        data = self.vec.transform(txt)
        return self.clf.predict_proba(data), self.get_weights(ds)