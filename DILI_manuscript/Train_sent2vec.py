from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from gensim.test.utils import datapath
import pandas as pd
import numpy as np
from nltk import word_tokenize
import os
import time
from sklearn.metrics import roc_curve, precision_recall_curve,auc, roc_auc_score, accuracy_score, precision_recall_curve, auc, f1_score
import matplotlib.pyplot as plt
from gensim import models
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#Load Bioword2vec model
import sent2vec
current_dir = os.getcwd()
print('Loading S2V model: ')
start = time.time()
model_path = 'BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
model = sent2vec.Sent2vecModel()
try: 
    model.load_model(model_path)
except Exception as e:
        print(e)
stop = time.time()
print('Loading S2V takes: ', -(start-stop))

def modelEval(y_true, y_preds, predict_probs):
    auroc = (roc_auc_score(y_true, predict_probs))
    accuracy = (accuracy_score(y_true, y_preds))
    f1 = f1_score(y_true, y_preds)
    precision, recall, threshold = precision_recall_curve(y_true, predict_probs)
    auprc = auc(recall, precision)
    return {"auroc": auroc, "accuracy": accuracy, "auprc": auprc, "f1_score": f1}


def getPredicts(clf, X):
    predict_probs = clf.predict_proba(X)[:, 1] #Predicted probability for the positive label
    predicts = clf.predict(X)
    return predicts, predict_probs


def retrieve_top_words(clf, vectorizer, top_k=5, top_positve_words=True):
    clf_name = clf.__class__.__name__
    if clf_name == 'LogisticRegression':
        coef_arr = np.array(clf.coef_).squeeze()
    elif clf_name == 'RandomForestClassifier':
        coef_arr = np.array(clf.feature_importances_).squeeze()
    else:
        raise (Exception('Classifier is not LR nor RF, cannot retrieve importance coef.'))

    Name_list = vectorizer.get_feature_names()
    if top_positve_words:
        # print('Retrieving Top '+str(top_k)+' words for positive samples')
        top_k_idx = coef_arr.argsort()[::-1][0:top_k]
    else:
        # print('Retrieving Top '+str(top_k)+' words for negative samples')
        top_k_idx = coef_arr.argsort()[0:top_k]
    top_k_words = []
    for idx in top_k_idx:
        top_k_words.append(Name_list[idx])
    # print(top_k_words)
    return (top_k_words)

def plot_ROC(y_true, y_pred, legend, lw):
    '''
    This function plots the ROC based on y_true and y_pred
    :param y_true: The ground truth of the samples
    :param y_pred: The predicted probablity of the samples
    :param legend: Legend of the curve
    :return: None
    '''
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auroc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = lw, label = legend + ' AUC=%0.3f' % auroc)

def plot_PRC(y_true, y_pred, legend, lw):
    '''
    This function plots the precision-recall curves based on y_true and y_pred
    :param y_true: The ground truth of the samples
    :param y_pred: The predicted probablity of the samples
    :param legend: Legend of the curve
    :return: None
    '''
    pre, rec, _ = precision_recall_curve(y_true, y_pred)
    auroc = auc(rec, pre)
    plt.plot(rec, pre, lw = lw, label = legend + ' AUC=%0.3f' % auroc)

class w2v_vectorizer():

    def __init__(self, wv):
        '''
        This functin initialize the word2vec model
        :param wv: a gensim word2vec model
        '''
        self.model = wv
        pass

    def transform(self,corpus):
        '''
        This function vectorize each document in the corpus
        :param corpus: pd.series of documents
        :return: feature: vectorized documents
        '''
        feature = np.empty((corpus.shape[0],200))
        sent_lst = list(corpus)
        print('W2V vectorization: get word vectors')
        vec_lst = [[self.model[word] for word in sent if word in self.model.vocab] for sent in tqdm(sent_lst)]
        print('W2V vectorization: get sentence vectors')
        for i in tqdm(list(range(feature.shape[0]))):
            feature[i,:] = np.mean(vec_lst[i],axis=0,keepdims=False) # continuous bag-of-word with average pooling
        return feature

class s2v_vectorizer():

    def __init__(self, sv):
        '''
        This functin initialize the word2vec model
        :param wv: a gensim word2vec model
        '''
        self.model = sv
        pass

    def transform(self,corpus):
        '''
        This function vectorize each document in the corpus
        :param corpus: pd.series of documents
        :return: feature: vectorized documents
        '''
        sent_lst = list(corpus)
        print('S2V vectorization: get word vectors')
        vec_lst = [self.model.embed_sentence(sent) for sent in tqdm(sent_lst)]
        print('S2V vectorization: get sentence vectors')
        feature = np.array(vec_lst).reshape(-1,700)
                   
        return feature
    
#Change here to choose corpus from STEM_TEXT or CLEAN_TEXT
os.chdir('/home/jupyter/tutorials/backup')
TEXT = 'CLEAN_TEXT'

#Load Dataset
Train_Data, Val_Data = pd.read_csv("Tokenized_train.csv"), pd.read_csv("Tokenized_val.csv")
sep_train, fusion_train = train_test_split(Train_Data, test_size = 0.25, shuffle=True, random_seed=9001)
sep_train_corpus, fusion_train_corpus, val_corpus = sep_train[TEXT], fusion_train[TEXT], Val_Data[TEXT]

#Preprocessing labels form boolean to int
Y_train_sep, Y_train_fusion, Y_val = sep_train['label'].astype(int), fusion_train['label'].astype(int), Val_Data['label'].astype(int)

#Instancing classifier (SVM and MLP take a lot time to run!)
clf_dict = {
    'LR':LogisticRegression(max_iter=500),'RF':RandomForestClassifier()
}

vectorizer_dict = {
    'S2V': s2v_vectorizer(sv=model)
}

#read-in previous training data
try:
    result_data = pd.read_csv('Pred_Data.csv')
    print('Found Previous Pred Data!')
except FileNotFoundError:
    result_data = pd.DataFrame({'y_val':Y_val})

try:
    result_dict = pd.read_csv('Result.csv')
    print('Found Previous Result Data!')
except FileNotFoundError:
    result_dict = pd.DataFrame({'Model' : [],'Vectorizer':[],'Corpus':[],'auroc': [], 'accuracy': [], 'auprc': [], 'f1_score': [],'Top_Positive_Words':[],'Top_Negative_Words':[]})
    


for vect_NAME in vectorizer_dict:
    vectorizer = vectorizer_dict[vect_NAME]

    #Fit vectorizer using train corpus
    print('S2V vectorizing: ')
    start = time.time()
    X_train = vectorizer.transform(train_corpus)
    np.save('S2V_STEM_train.npy',X_train)
    X_val = vectorizer.transform(val_corpus)
    np.save('S2V_STEM_val.npy', X_val)
    stop = time.time()
    print('S2V vectorization takes: ', (stop-start))

    #Fit clf
    for clf_NAME in clf_dict:
        clf = clf_dict[clf_NAME]
        print('Fitting Mode: '+vect_NAME+' + ' + clf_NAME +' + ' + TEXT)

        start = time.time()
        clf.fit(X_train,Y_train)
        stop = time.time()
        print('Fitting model takes: ', (stop - start))
        preds, pred_probs = getPredicts(clf, X_val)
        result = modelEval(Y_val, preds, pred_probs)
        result['Model']=clf_NAME
        result['Vectorizer']=vect_NAME
        result['Corpus']=TEXT
        result['Top_Positive_Words']='None'
        result['Top_Negative_Words']='None'
        result_dict = result_dict.append(result,ignore_index=True)

        #save_file
        y_pred_Name = TEXT +'_'+ vect_NAME + '_' + clf_NAME
        y_prob_Name = 'Prob_' + y_pred_Name

        result_data[y_pred_Name] = preds
        result_data[y_prob_Name] = pred_probs

result_data.to_csv('Pred_Data.csv',index=False)
result_dict.to_csv('Result.csv',index=False)
#print(result_dict)
