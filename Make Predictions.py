# This file do the prediction steps for the text. Created by FW
# Specific steps: Read from processed data/ Load trained model from file/ Pred

### --- LOAD DEPENDENCIES --- ###
from sklearn.model_selection import train_test_split
from libs.utils import *
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import gensim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--FilePath', type=str, default='Data/Example.tsv', help='Set path to the pred/training data')
parser.add_argument('--Mode', type=str, default='PRED', help='Mode: PRED (default) for making predictions with pretrained model or \n TRAIN_EVAL for training a new model')
parser.add_argument('--Text', type=str, default='STEM_TEXT', help='Choose from STEM_TEXT (default) of CLEAN_TEXT')

args = parser.parse_args()



### --- PREPROCESSING FUNCTIONS --- ###
def preprocess_text(df):
    '''
    This function removes weird characters and punctuations
    :param df: raw df
    :return:clean_df: with weird characters removed
    '''
    df['STEM_TEXT'] = 'nan'
    for i in range(df.shape[0]):
        df['CLEAN_TEXT'][i] = gensim.parsing.preprocessing.strip_tags(df['CLEAN_TEXT'][i])
        df['CLEAN_TEXT'][i] = gensim.parsing.preprocessing.strip_punctuation(df['CLEAN_TEXT'][i])
        df['CLEAN_TEXT'][i] = gensim.parsing.preprocessing.strip_multiple_whitespaces(df['CLEAN_TEXT'][i])
        df['CLEAN_TEXT'][i] = gensim.parsing.preprocessing.strip_numeric(df['CLEAN_TEXT'][i])
        df['CLEAN_TEXT'][i] = gensim.parsing.preprocessing.remove_stopwords(df['CLEAN_TEXT'][i])
        df['STEM_TEXT'][i] = gensim.parsing.preprocessing.stem_text(df['CLEAN_TEXT'][i])
    return df

### --- INPUT THE FILE NAME --- ###
FILE = args.FilePath # Input the name of the user's file that include the PubMed Title and/or Abstract
if '.tsv' in FILE:
    delimiter = '\t'
else:
    delimiter = ','

### --- SPECIFY PARAMS --- ###
MODEL_NAME = 'TFIDF_LR_BEST_80%'

# STEM_TEXT or CLEAN_TEXT
TEXT = args.Text

# 'PRED' or 'TRAIN_EVAL'
MODE = args.Mode

### --- READ DATA --- ###
#Load Dataset
if MODE =='TRAIN_EVAL':
    print("Mode: Train and Evaluate")
    Pred_Data = pd.read_csv(FILE, delimiter = delimiter)
    Pred_Data['TEXT'] = Pred_Data['Title'].astype(str) + ' ' + Pred_Data['Abstract'].astype(str) # Concatenate the strings of the title and the abstract

    ### --- PREPROCESSING --- ###
    Pred_Data['CLEAN_TEXT'] = Pred_Data['TEXT'].str.lower()
    Pred_Data = preprocess_text(Pred_Data)

    ### --- STEM/CLEAN TEXT --- ###
    pred_corpus= Pred_Data[TEXT]
    
    Train_Data, Val_Data = train_test_split(pred_corpus,test_size=0.2,random_state=42)
    train_corpus, val_corpus= Train_Data[TEXT], Val_Data[TEXT]

    #Preprocessing labels form boolean to int
    Y_train, Y_val = Train_Data['label'].astype(int), Val_Data['label'].astype(int)  
    
    #Text Vectorization
    vectorizer = text.TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_corpus)
    X_val = vectorizer.transform(val_corpus)  

    #Instancing classifier
    print("Model:"+MODEL_NAME)
    clf = {
        'TFIDF_LR_BEST_80%':LogisticRegression(max_iter=1000,penalty='l2',C=10,class_weight={1:1,0:1}),
        }[MODEL_NAME]

    #Train and Evaluate
    clf.fit(X_train,Y_train)
    preds, pred_probs = getPredicts(clf, X_val)
    result = modelEval(Y_val, preds, pred_probs)

    print('Prediction Performance:')
    print(pd.DataFrame.from_dict(result,orient='index').transpose())


elif MODE=='PRED':
    print("Mode: prediction")

    Pred_Data = pd.read_csv(FILE, delimiter = delimiter)
    Pred_Data['TEXT'] = Pred_Data['Title'].astype(str) + ' ' + Pred_Data['Abstract'].astype(str) # Concatenate the strings of the title and the abstract

    ### --- PREPROCESSING --- ###
    Pred_Data['CLEAN_TEXT'] = Pred_Data['TEXT'].str.lower()
    Pred_Data = preprocess_text(Pred_Data)

    ### --- STEM/CLEAN TEXT --- ###
    pred_corpus= Pred_Data[TEXT]

    ### --- LOAD MODEL --- ###
    # Load from file
    with open('Model/'+MODEL_NAME+'.pkl', 'rb') as file:
        model = pickle.load(file)
        vectorizer = model['vectorizer']
        clf =model['clf']
        print("Model Loaded:"+MODEL_NAME)


    ### ---- LOAD A for Conformal Prediction --- ###
    if MODEL_NAME != 'TFIDF_LR_BEST_80%':
        raise KeyError('Model not correct! Use TFIDF_LR_BEST_80 for conformal prediction!')

    A_pred_prob_full=np.load('Model/A_pred_prob_full.npy')

    #if label=0, a=pred_prob, if label=1, a=1-pred_prob (pred_prob is the probabily of label=1 given by LR)
    A_train_full_y0 = np.copy(A_pred_prob_full)
    A_train_full_y1 = np.copy(1-A_pred_prob_full)

    #Text Vectorization
    X_pred = vectorizer.transform(pred_corpus)

    ### --- PREDICT --- ###
    preds, pred_probs = getPredicts(clf, X_pred)

    #Calculate A_pred for y=0 and y=1
    A_pred_y0 = np.copy(pred_probs)
    A_pred_y1 = np.copy(1-pred_probs)

    #Calculate p_value
    pred_p_value = np.zeros((len(preds),2))
    for m in range(len(preds)):
        cnt0=len(A_train_full_y0[A_train_full_y0 >= A_pred_y0[m]])
        pred_p_value[m][0] = (cnt0 + 1) / (len(A_train_full_y0) + 1)

        cnt1=len(A_train_full_y1[A_train_full_y1 >= A_pred_y1[m]])
        pred_p_value[m][1] = (cnt1 + 1) / (len(A_train_full_y1) + 1)

    #Calculate credibility
    credibility= np.max(pred_p_value,axis=1)

    result_pred = pd.DataFrame({'Pred':preds,'Pred_Probability':pred_probs,'Credibility':credibility})
    result_pred.to_csv('Result/Pred.csv',index=False)
    print("Predictions saved to: Result/Pred.csv")

else:
    print("Wrong Mode")