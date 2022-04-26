from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc,f1_score
import numpy as np
import pandas as pd
from sklearn.utils import resample

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

def retrieve_top_words(clf,vectorizer,top_k=5,top_positve_words=True):
    clf_name = clf.__class__.__name__
    if clf_name == 'LogisticRegression':
        coef_arr = np.array(clf.coef_).squeeze()
    elif  clf_name == 'RandomForestClassifier':
        coef_arr = np.array(clf.feature_importances_).squeeze()
    else:
        raise(Exception('Classifier is not LR nor RF, cannot retrieve importance coef.'))
    
    Name_list = vectorizer.get_feature_names()
    if top_positve_words:
        #print('Retrieving Top '+str(top_k)+' words for positive samples')
        top_k_idx=coef_arr.argsort()[::-1][0:top_k]
    else:
        #print('Retrieving Top '+str(top_k)+' words for negative samples')
        top_k_idx=coef_arr.argsort()[0:top_k]
    top_k_words =[]
    for idx in top_k_idx:
        top_k_words.append(Name_list[idx])
    #print(top_k_words)
    return(top_k_words)

def ErrorAnalysis(preds,GT_label,GT_text):
    '''
    Return DataFrame of FN and FP

    args:
    preds: predicted labels
    GT_label: ground truth labels
    val_text: original text before vectorization

    output:
    Error_DF: DataFrame that contains FN text and index, FP text and index
    '''
    False_Neg = []
    False_Pos = []
    False_Neg_Idx = []
    False_Pos_Idx = []
    for idx in range(preds.shape[0]):
        if preds[idx] == 1 and GT_label[idx] == 0:
            False_Pos.append(GT_text[idx])
            False_Pos_Idx.append(idx)
        elif preds[idx] == 0 and GT_label[idx] == 1:
            False_Neg.append(GT_text[idx])
            False_Neg_Idx.append(idx)
    Error_dict = {'False_Positive':False_Pos,'False_Negative':False_Neg,'False_Pos_Idx':False_Pos_Idx,'False_Neg_Idx':False_Neg_Idx}
    Error_DF = pd.DataFrame.from_dict(Error_dict,orient='index').transpose()
    return Error_DF

def bootstrap(model, x,y, times = 2000):
    '''
    Perform bootstrapping, store results for further analysis and visualization
    :param x_train: training set X
    :param y_train: training set Y
    :param x_test: testing set X
    :param y_test: testing set Y
    :param featrue_eng: feature engineering method list to pass in feature engineering function
    :param times: how many times to bootstrap
    :return: dictionary of metrics, dict['<metric name>'] = [<values, length = fold>]
    '''
    results = []
    index = np.arange(x.shape[0])
    for i in range(times):
        boot_index = resample(index, replace=True, n_samples=None, random_state=9001+i)
        x_boot, y_boot = x[boot_index], y[boot_index]
        model.fit(x_boot,y_boot)
        results.append(model.coef_[0])
    return results