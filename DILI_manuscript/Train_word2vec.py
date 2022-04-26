from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pandas as pd
import numpy as np
from libs.utils import *
from nltk import word_tokenize
import os
import time

#Load Bioword2vec model
current_dir = os.getcwd()
print('Loading W2V model: ')
start = time.time()
#wv1 = KeyedVectors.load_word2vec_format(datapath(current_dir+'\\wikipedia-pubmed-and-PMC-w2v.bin'),binary=True)
wv2 = KeyedVectors.load_word2vec_format(datapath(current_dir+'/models/BioWordVec_PubMed_MIMICIII_d200.vec.bin'),binary=True)
stop = time.time()
print('Loading W2V takes: ', -(start-stop))
print(len(wv2.vocab))

# #Change here to choose corpus from STEM_TEXT or CLEAN_TEXT
# TEXT = 'STEM_TEXT'

# #Load Dataset
# Train_Data, Val_Data = pd.read_csv("Tokenized_train.csv"), pd.read_csv("Tokenized_val.csv")
# tokenized_train = Train_Data['TOKENS']
# tokenized_val = Val_Data['TOKENS']

# #Preprocessing labels form boolean to int
# Y_train, Y_val = Train_Data['label'].astype(int), Val_Data['label'].astype(int)

# #Instancing classifier (SVM and MLP take a lot time to run!)
# clf_dict = {
#     'LR':LogisticRegression(max_iter=500),'RF':RandomForestClassifier()
# }

# #Instancing vectorizer
# # vectorizer_dict = {
# #     'W2V1': w2v_vectorizer(wv=wv1)
# # }

# vectorizer_dict = {
#     'W2V2': w2v_vectorizer(wv=wv2)
# }

# #read-in previous training data
# try:
#     result_data = pd.read_csv('Pred_Data.csv')
#     print('Found Previous Pred Data!')
# except FileNotFoundError:
#     result_data = pd.DataFrame({'y_val':Y_val})

# try:
#     result_dict = pd.read_csv('Result.csv')
#     print('Found Previous Result Data!')
# except FileNotFoundError:
#     result_dict = pd.DataFrame({'Model' : [],'Vectorizer':[],'Corpus':[],'auroc': [], 'accuracy': [], 'auprc': [], 'f1_score': [],'Top_Positive_Words':[],'Top_Negative_Words':[]})



# for vect_NAME in vectorizer_dict:
#     vectorizer = vectorizer_dict[vect_NAME]

#     #Fit vectorizer using train corpus
#     print('W2V vectorizing: ')
#     start = time.time()
#     X_train = vectorizer.transform(tokenized_train)
#     np.save('W2V2_STEM_train.npy',X_train)
#     X_val = vectorizer.transform(tokenized_val)
#     np.save('W2V2_STEM_val.npy', X_val)
#     stop = time.time()
#     print('W2V vectorization takes: ', (stop-start))

#     #Fit clf
#     for clf_NAME in clf_dict:
#         clf = clf_dict[clf_NAME]
#         print('Fitting Mode: '+vect_NAME+' + ' + clf_NAME +' + ' + TEXT)

#         start = time.time()
#         clf.fit(X_train,Y_train)
#         stop = time.time()
#         print('Fitting model takes: ', (stop - start))
#         preds, pred_probs = getPredicts(clf, X_val)
#         result = modelEval(Y_val, preds, pred_probs)
#         result['Model']=clf_NAME
#         result['Vectorizer']=vect_NAME
#         result['Corpus']=TEXT
#         result['Top_Positive_Words']='None'
#         result['Top_Negative_Words']='None'
#         result_dict = result_dict.append(result,ignore_index=True)

#         #save_file
#         y_pred_Name = TEXT +'_'+ vect_NAME + '_' + clf_NAME
#         y_prob_Name = 'Prob_' + y_pred_Name

#         result_data[y_pred_Name] = preds
#         result_data[y_prob_Name] = pred_probs

# result_data.to_csv('Pred_Data.csv',index=False)
# result_dict.to_csv('Result.csv',index=False)
#print(result_dict)

