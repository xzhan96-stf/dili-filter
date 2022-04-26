# DILI-filter
## Reliably Filter Drug-induced Liver Injury Literature with Natural Language Processing and Conformal Prediction

<details>
<summary>Abstract</summary>
Drug-induced liver injury describes the adverse effects of drugs that damage liver. Life-threatening results including liver failure or death were also reported in severe cases. Therefore, the events related to liver injury are strictly monitored for all approved drugs and the liver toxicity is an important assessments for new drug candidates. These reports are documented in research papers that contain preliminary in vitro and in vivo experiments. Conventionally, data extraction from previous publications relies heavily on resource-demanding manual labelling, which considerably restricted the efficiency of the information extraction. The development of natural language processing techniques enabled the automatic processing of biomedical texts. Herein, based on around 28,000 papers (titles and abstracts) provided by the Critical Assessment of Massive Data Analysis challenge, this study benchmarked model performances on filtering out liver-damage-related literature. Among five text embedding techniques, the model using term frequency-inverse document frequency (TFIDF) and logistic regression outperformed others with an accuracy of 0.957 on the validation set. Furthermore, an ensemble model with similar overall performances was developed with a logistic regression model on the predicted probability given by separate models with different vectorization techniques. The ensemble model achieved a high accuracy of 0.954 and an F1 score of 0.955 in the hold-out validation data provided by the challenge. Moreover, important words in positive/negative predictions were identified via model interpretation. The prediction reliability was quantified with conformal prediction, which provides users with a control over the prediction uncertainty. Overall, the ensemble model and TF-IDF model reached satisfactory classification results, which can be further used by researchers to rapidly filter literature that describes events related to liver injury induced by medications.
</details>



## General purpose
Predict whether a publication is related to drug-induced liver injury or not based on the publication title and/or abstract, with prediction credibility given. Low credibility (<0.65) indicates that the prediction is not reliable and manual check may be necessary to avoid false predictions.


## How to use it to make prediction?
1. Install the required packages listed in the requirements.txt, including gensim, numpy, pandas, scikit-learn, regex;
2. Prepare the publication titles and/or abstracts in .csv or .tsv files (example format in ./Data);
3. Run Make_Predictions.py with the args: *--FilePath [YOUR FILE LOCATION]* or modify the default Example.tsv file in the ./Data folder
4. Harvest the predictions in the ./Result folder.

## Supplementary Code in ./DILI_manuscript
These are the example codes related to the experiments done in the manuscript entitled "Reliably Filter Drug-induced Liver Injury Literature with Natural Language Processing and Conformal Prediction" for reproducibility purposes. These codes are not necessary for the application of DILI filter in real-world scenarios.
