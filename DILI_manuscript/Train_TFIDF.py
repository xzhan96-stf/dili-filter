from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc,f1_score

def modelEval(y_true, y_preds, predict_probs):
    auroc = (roc_auc_score(y_true, predict_probs))
    accuracy = (accuracy_score(y_true, y_preds))
    f1 = f1_score(y_true, y_preds)
    precision, recall, threshold = precision_recall_curve(y_true, predict_probs)
    auprc = auc(recall, precision)
    return {"auroc": auroc, "accuracy": accuracy, "auprc": auprc,"f1_score":f1}

def getPredicts(clf, X):
    predict_probs = clf.predict_proba(X)[:, 1]
    predicts = clf.predict(X)
    return predicts, predict_probs

#Change here to choose corpus from STEM_TEXT or CLEAN_TEXT
TEXT = 'STEM_TEXT'

#Load Dataset
Train_Data, Val_Data = pd.read_csv("Data/train.csv"),pd.read_csv("Data/val.csv")
train_corpus, val_corpus= Train_Data[TEXT], Val_Data[TEXT]

#Preprocessing labels form boolean to int
Y_train, Y_val = Train_Data['label'].astype(int), Val_Data['label'].astype(int)

holdout_val = pd.read_csv('Data/AdditionalDILItest.csv')
holdout_val_corpus = holdout_val[TEXT]

#Instancing classifier (SVM and MLP take a lot time to run!)
clf_dict = {
    'LR_BEST':LogisticRegression(max_iter=1000,penalty='l2',C=10,class_weight={1:1,0:1}),
}

#Instancing vectorizer
vectorizer_dict = {
    #'BOW':text.CountVectorizer(),
    'TFIDF':text.TfidfVectorizer()
}


#read-in previous training data
try:
    result_data = pd.read_csv('Result/Pred_Data.csv')
    print('Found Previous Pred Data!')
except FileNotFoundError:
    result_data = pd.DataFrame({'y_val':Y_val})

try:
    result_dict = pd.read_csv('Result/Result.csv')
    print('Found Previous Result Data!')
except FileNotFoundError:
    result_dict = pd.DataFrame({'Comments': [],'Model' : [],'Vectorizer':[],'Corpus':[],'auroc': [], 'accuracy': [], 'auprc': [], 'f1_score': [],'Top_Positive_Words':[],'Top_Negative_Words':[]})



for vect_NAME in vectorizer_dict:
    vectorizer = vectorizer_dict[vect_NAME]

    #Fit vectorizer using train corpus
    X_train = vectorizer.fit_transform(train_corpus)
    X_val = vectorizer.transform(val_corpus)

    #Fit clf
    for clf_NAME in clf_dict:
        clf = clf_dict[clf_NAME]
        print('Fitting Mode: '+vect_NAME+' + ' + clf_NAME +' + ' + TEXT)

        clf.fit(X_train,Y_train)
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

result_data.to_csv('Result/Pred_Data.csv',index=False)
result_dict.to_csv('Result/Result.csv',index=False)
print(result_dict)
print(result)
