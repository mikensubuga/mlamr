# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.svm import SVC,LinearSVC 
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import matthews_corrcoef,auc, roc_curve,plot_roc_curve, plot_precision_recall_curve,classification_report, confusion_matrix,average_precision_score, precision_recall_curve
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import imblearn
from collections import Counter
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV





# %%
#Load data from csv file
#data = pd.read_csv('merge_phenogeno.csv')
phenodata = pd.read_csv('PhenoData.csv')
phenodata.shape



# %%
phenodata.head(5)

# %%
#Load genotype data
#genodata = pd.read_csv('snps_input_2028.csv')
genodata = pd.read_csv('merged_ten.csv')

genodata.shape

# %%
genodata.head(5)

# %%
#Create a new dataframe and encode A,G,C,T to 1,2,3,4 and also the NaN values to 0
genodata1 = genodata.copy()
genodata1 = genodata1.replace('A',1)
genodata1 = genodata1.replace('G',2)
genodata1 = genodata1.replace('C',3)
genodata1 = genodata1.replace('T',4)
genodata1 = genodata1.replace(np.nan,0)

genodata1.head(5)

# %%
#merge phenotype and genotype data by prename
data = pd.merge(phenodata, genodata1, on='prename')


# %%
data.head(5)

# %%
#Count number of categories in the data for row CIP
data['CIP'].value_counts()

# %%
#Delete row where CIP is equal to I
data = data[data.CIP != 'I']

# %%
data['CIP'].value_counts()

# %%
#Data shape
data.shape

# %%


# %%
#### Load the independent dataset for Uganda
Ugandaphenodata = pd.read_csv('phenoCIPUganda.csv')
Ugandagenodata = pd.read_csv('snps_input_Uganda.csv')

# %%
#merge phenotype and genotype data by prename
Ugandadata = pd.merge(Ugandaphenodata, Ugandagenodata, on='prename')
Ugandadata.head(3)


# %%
Ugandadata['CIP'].value_counts()

# %%
#Remove relevant columns/features for both datasets
data = data.drop(['prename'], axis=1)
Ugandadata = Ugandadata.drop(['prename'], axis=1)


# %%


# %%


# %%


# %%
data['CIP'] = data['CIP'].map({'S': 0, 'R': 1})
Ugandadata['CIP'] = Ugandadata['CIP'].map({'S': 0, 'R': 1})


# %%
#Drop the rows with missing values
data = data.dropna()
Ugandadata = Ugandadata.dropna()


# %%
# Find common columns
common_cols = data.columns.intersection(Ugandadata.columns)
X = data[common_cols]
Uganda_X = Ugandadata[common_cols]

# %%
#Encode categorical phenotype data
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#data['CIP'] = le.fit_transform(data['CIP'])
#Ugandadata['CIP'] = le.fit_transform(Ugandadata['CIP'])


# %%
y = X['CIP']
Uganda_y = Uganda_X['CIP']


# %%
X = X.drop(['CIP'], axis=1)
Uganda_X = Uganda_X.drop(['CIP'], axis=1)

# %%
#Convert X to int64
X = X.astype('int64')
y = y.astype('int64')

# %%


# %%


# %%


# %%
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_keras_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=False),
    'Feed-Forward NN (Keras)': KerasClassifier(build_fn=create_keras_model, epochs=20, batch_size=32, verbose=0)
}


# %%


# %%


# %%


# %%
#Function to evaluate the models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(classifiers, X_train, y_train, X_test, y_test):
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        auc_roc = roc_auc_score(y_test, y_pred_proba_test)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
    return results

results = evaluate_models(classifiers, X_train, y_train, X_test, y_test)




# %%
results 

# %%


# %%
df = pd.DataFrame(results).transpose().round(2)
print(df)

# %%
df

# %%
#Export to CSV
df.to_csv('results_table_CIP.csv')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(results, label):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(results.keys()), y=[res[metric] for res in results.values()])
        plt.title(f'{metric.capitalize()} for Different Models on {label} Data')
        #turn 45 degrees labels for x axis
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.show()

plot_metrics(results, "Imbalanced CIP")


# %%


# %%


# %%
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(results, X_test, y_test, label):
    plt.figure(figsize=(10, 10))
    for name, clf in classifiers.items():
        y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves for Different Models for {} Data'.format(label))
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curves(results, X_test, y_test, "Imbalanced CIP")



# %%


# %%


# %%
#Perform k-fold cross-validation to obtain AUC scores for each model:
from sklearn.model_selection import cross_val_score

def cross_val_auc(classifiers, X, y, cv=5):
    auc_scores = {}
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        auc_scores[name] = scores
        
    return auc_scores

auc_scores = cross_val_auc(classifiers, X, y)


# %%


# %%
auc_scores

# %%


# %%
import numpy as np

def plot_auc_boxplot_violinplot(auc_scores, label):
    model_names = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    fig, axes = plt.subplots(figsize=(10, 6))
    
    # Violin plot
    sns.violinplot(data=scores, ax=axes, inner='quartile')
    axes.set_title('Violin Plot of AUC Scores of different models for {} Data'.format(label))
    axes.set_xticklabels(model_names, fontsize=8, rotation=45) # Set font size to 10
    axes.set_xticklabels(model_names)
    axes.set_ylabel('AUC Score')
    
    plt.show()

plot_auc_boxplot_violinplot(auc_scores, "Imbalanced CIP")

# %%


# %%
auc_scores

# %%
#Calculate the statistical significance of the difference in AUC scores between the models:
from scipy.stats import ttest_ind

def auc_statistical_significance(auc_scores):
    model_names = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            print(f'{model_names[i]} vs {model_names[j]}:')
            print(ttest_ind(scores[i], scores[j]))
            print()

auc_statistical_significance(auc_scores)

    

# %%
#Use a multiple comparison test to determine which models are significantly different from each other:
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def auc_multiple_comparison_test(auc_scores):
    model_names = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    # Flatten the scores into a single list
    scores_flat = [score for model_scores in scores for score in model_scores]
    
    # Create a list of model names repeated for each score
    model_names_flat = [[model_name] * len(scores[i]) for i, model_name in enumerate(model_names)]
    model_names_flat = [model_name for model_names in model_names_flat for model_name in model_names]
    
    # Perform multiple comparison test
    tukey_results = pairwise_tukeyhsd(scores_flat, model_names_flat)
    print(tukey_results)

auc_multiple_comparison_test(auc_scores)

# %%
import pandas as pd

def auc_multiple_comparison_test_csv(auc_scores, output_file='tukey_results_CIP.csv'):
    model_names = list(auc_scores.keys())
    scores = list(auc_scores.values())
    
    # Flatten the scores into a single list
    scores_flat = [score for model_scores in scores for score in model_scores]
    
    # Create a list of model names repeated for each score
    model_names_flat = [[model_name] * len(scores[i]) for i, model_name in enumerate(model_names)]
    model_names_flat = [model_name for model_names in model_names_flat for model_name in model_names]
    
    # Perform multiple comparison test
    tukey_results = pairwise_tukeyhsd(scores_flat, model_names_flat)
    print(tukey_results)
    
    # Convert Tukey HSD results to a pandas DataFrame
    tukey_results_df = pd.DataFrame(tukey_results._results_table.data[1:], columns=tukey_results._results_table.data[0])
    
    # Save the DataFrame to a CSV file
    tukey_results_df.to_csv(output_file, index=False)

auc_multiple_comparison_test_csv(auc_scores)



# %%
def print_mean_auc_scores(auc_scores):
    print("Mean AUC scores:")
    for model_name, scores in auc_scores.items():
        mean_auc = np.mean(scores)
        print(f"{model_name}: {mean_auc:.4f}")

print_mean_auc_scores(auc_scores)


# %%
def logistic_regression_feature_importance_50(lr_model, feature_names):
    importance = np.abs(lr_model.coef_[0])
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]

# %%
def logistic_regression_feature_importance(lr_model, feature_names):
    importance = np.abs(lr_model.coef_[0])
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]


# %%
def random_forest_feature_importance(rf_model, feature_names):
    importance = rf_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]


# %%
from sklearn.feature_selection import RFE

#def svm_feature_importance(svm_model, feature_names, X_train, y_train):
 #   if svm_model.kernel == 'linear':
  #      importance = np.abs(svm_model.coef_[0])
  #      feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
   # else:
    #    selector = RFE(svm_model, n_features_to_select=15)
     #   selector.fit(X_train, y_train)
      #  feature_importance = sorted(zip(feature_names, selector.ranking_), key=lambda x: x[1])
    
    #return feature_importance[:10]


# %%
from sklearn.feature_selection import SelectKBest, chi2

def svm_feature_importance(svm_model, feature_names, X_train, y_train):
    if svm_model.kernel == 'linear':
        importance = np.abs(svm_model.coef_[0])
        feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    else:
        selector = SelectKBest(chi2, k=15)
        selector.fit(X_train, y_train)
        feature_importance = sorted(zip(feature_names, selector.scores_), key=lambda x: x[1], reverse=True)
    
    return feature_importance[:50]


# %%
def gradient_boosting_feature_importance(gb_model, feature_names):
    importance = gb_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]


# %%
def xgboost_feature_importance(xgb_model, feature_names):
    importance = xgb_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]

def lightgbm_feature_importance(lgb_model, feature_names):
    importance = lgb_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]

def catboost_feature_importance(cb_model, feature_names):
    importance = cb_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]

def keras_nn_feature_importance(keras_model, feature_names):
    weights = np.abs(keras_model.model.layers[0].get_weights()[0]).mean(axis=1)
    feature_importance = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)
    return feature_importance[:50]


# %%


# %%


# %%


# %%


# %%


# %%
lr_importance_50 = logistic_regression_feature_importance_50(classifiers['Logistic Regression'], feature_names)
print(lr_importance_50)

# %%
#Add lr_importance_50 to a table
lr_importance_50 = pd.DataFrame(lr_importance_50, columns=['feature', 'importance'])
lr_importance_50

# %%
feature_names = X.columns
lr_importance = logistic_regression_feature_importance(classifiers['Logistic Regression'], feature_names)
rf_importance = random_forest_feature_importance(classifiers['Random Forest'], feature_names)
gb_importance = gradient_boosting_feature_importance(classifiers['Gradient Boosting'], feature_names)
xgb_importance = xgboost_feature_importance(classifiers['XGBoost'], feature_names)
lgb_importance = lightgbm_feature_importance(classifiers['LightGBM'], feature_names)
cb_importance = catboost_feature_importance(classifiers['CatBoost'], feature_names)
keras_importance = keras_nn_feature_importance(classifiers['Feed-Forward NN (Keras)'], feature_names)
svm_importance = svm_feature_importance(classifiers['SVM'], feature_names, X_train, y_train)


#keras_importance = keras_nn_feature_importance(classifiers['Keras Neural Network'], feature_names)

print("Top 10 features for Logistic Regression:\n", lr_importance)
print("\nTop 10 features for Random Forest:\n", rf_importance)
print("\nTop 10 features for Gradient Boosting:\n", gb_importance)
print("\nTop 10 features for XGBoost:\n", xgb_importance)
print("\nTop 10 features for LightGBM:\n", lgb_importance)
print("\nTop 10 features for CatBoost:\n", cb_importance)
print("\nTop 10 features for Keras Neural Network:\n", keras_importance)
print("\nTop 10 features for SVM:\n", svm_importance)


#print("\nTop 10 features for Keras Neural Network:\n", keras_importance)


# %%
importances_list = [lr_importance, rf_importance, gb_importance, xgb_importance, lgb_importance, cb_importance, keras_importance, svm_importance]
model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Feed-Forward NN (Keras)", "SVM"]


# %%


# %%
from tabulate import tabulate

def draw_top_features_table(importances_list, model_names):
    feature_importance_data = {}

    for model_name, importances in zip(model_names, importances_list):
        features_scores = [(f"{i+1}. {feature}", round(score, 2)) for i, (feature, score) in enumerate(importances[:50])]
        feature_importance_data[model_name] = features_scores

    feature_importance_df = pd.DataFrame(feature_importance_data)
    headers = [('Feature', 'Score')] * len(model_names)
    header_tuples = list(zip(model_names, headers))
    multi_header = pd.MultiIndex.from_tuples(header_tuples)
    feature_importance_df.columns = multi_header
    return tabulate(feature_importance_df, headers='keys', tablefmt='psql', showindex=False)

# Call the function to draw the table
top_features_table = draw_top_features_table(importances_list, model_names)
print(top_features_table)


# %%
import pandas as pd

def draw_top_features_table(importances_list, model_names):
    feature_importance_data = {}

    for model_name, importances in zip(model_names, importances_list):
        features_scores = [(f"{i+1}. {feature}", round(score, 2)) for i, (feature, score) in enumerate(importances[:10])]
        feature_importance_data[model_name] = features_scores

    feature_importance_df = pd.DataFrame(feature_importance_data)
    headers = [('Feature', 'Score')] * len(model_names)
    header_tuples = list(zip(model_names, headers))
    multi_header = pd.MultiIndex.from_tuples(header_tuples)
    feature_importance_df.columns = multi_header

    return feature_importance_df

# Call the function to draw the table
top_features_df = draw_top_features_table(importances_list, model_names)
print(top_features_df)


# %%
importances

# %%
top_features_df.to_csv('top_features.csv') 

# %%
lr_importance

# %%
import pandas as pd

def draw_top_features_table(importances_list, model_names):
    feature_importance_data = {}

    for model_name, importances in zip(model_names, importances_list):
        features_scores = [(f"{i+1}. {feature}", round(score, 2)) for i, (feature, score) in enumerate(importances[:10])]
        feature_importance_data[model_name] = features_scores

    feature_importance_df = pd.DataFrame(feature_importance_data)
    headers = [('Feature', 'Score')] * len(model_names)
    header_tuples = list(zip(model_names, headers))
    multi_header = pd.MultiIndex.from_tuples(header_tuples)
    feature_importance_df.columns = multi_header

    feature_list = [feature for importances in importances_list for feature, _ in importances[:10]]
    repeated_features = [feature for feature in set(feature_list) if feature_list.count(feature) > 1]

    def highlight_repeated_features(val):
        if isinstance(val, str) and any(val.endswith(f" {feature}") for feature in repeated_features):
            return 'background-color: red'
        else:
            return ''

    styled_df = feature_importance_df.style.applymap(highlight_repeated_features)
    return styled_df

# Call the function to draw the table
top_features_styled_df = draw_top_features_table(importances_list, model_names)
top_features_styled_df


# %%


# %%


# %%


# %%


# %%


# %%
def plot_feature_importances(importances, model_name):
    features, scores = zip(*importances)
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, scores, align='center', alpha=0.5)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title(f'Top 10 Features for {model_name} on imbalanced CIP')
    
    plt.gca().invert_yaxis()  # To show features from top to bottom
    plt.show()

plot_feature_importances(lr_importance, 'Logistic Regression')
plot_feature_importances(rf_importance, 'Random Forest')
plot_feature_importances(gb_importance, 'Gradient Boosting')
plot_feature_importances(xgb_importance, 'XGBoost')
plot_feature_importances(lgb_importance, 'LightGBM')
plot_feature_importances(cb_importance, 'CatBoost')
plot_feature_importances(keras_importance, 'Feed Foward Neural Network')
plot_feature_importances(svm_importance, 'SVM')



# %%


# %%


# %%


# %%


# %% [markdown]
# Omit Normalisation

# %%
def normalize_importance_scores(importances):
    features, scores = zip(*importances)
    max_score = max(scores)
    normalized_scores = [score / max_score for score in scores]
    return list(zip(features, normalized_scores))

lr_normalized = normalize_importance_scores(lr_importance)
rf_normalized = normalize_importance_scores(rf_importance)
gb_normalized = normalize_importance_scores(gb_importance)

def plot_grouped_feature_importances(importances_list, model_names):
    n_models = len(importances_list)
    n_features = len(importances_list[0])
    
    bar_width = 0.2
    index = np.arange(n_features)
    
    plt.figure(figsize=(12, 8))
    
    for i, (importances, model_name) in enumerate(zip(importances_list, model_names)):
        _, scores = zip(*importances)
        plt.bar(index + i * bar_width, scores, bar_width, label=model_name)
    
    plt.xticks(index + bar_width * (n_models - 1) / 2, [imp[0] for imp in importances_list[0]])
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance Score')
    plt.title('Top 10 Features for Each Model')
    plt.legend()
    plt.show()

plot_grouped_feature_importances(
    [lr_normalized, rf_normalized, gb_normalized],
    ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
)


# %%


# %%


# %%
X_train.shape

# %%
X_train.head()

# %%


# %%


# %%
lr_importance

# %%
###Interaction analysis
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Extract the feature names from the lr_importance list and convert them to integers
top_features = [int(feature) for feature, importance in lr_importance]

# Convert X_train and X_test to DataFrames
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

# Find the indices of the top features in the original dataset
top_feature_indices = [X_train_df.columns.get_loc(str(feature)) for feature in top_feature_names]

# Create a dataset with only the top important features using the feature indices
X_train_top = X_train_df.iloc[:, top_feature_indices]
X_test_top = X_test_df.iloc[:, top_feature_indices]



# Create interaction terms
from sklearn.preprocessing import PolynomialFeatures
interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_interaction = interaction.fit_transform(X_train_top)
X_test_interaction = interaction.transform(X_test_top)

# Train a model with interaction terms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logreg_interaction = make_pipeline(StandardScaler(), LogisticRegression())
logreg_interaction.fit(X_train_interaction, y_train)

# Extract feature importances (coefficients) for the interaction terms
coef_interaction = logreg_interaction.named_steps['logisticregression'].coef_
interaction_importances = np.abs(coef_interaction)

# Analyze the importance of interaction terms
interaction_terms = interaction.get_feature_names_out(input_features=X_train_top.columns)
important_interactions = sorted(zip(interaction_terms, interaction_importances[0]), key=lambda x: -x[1])


# %%
important_interactions

# %%
import matplotlib.pyplot as plt

# Select the top 10 interaction terms
top_n = 15
top_interactions = important_interactions[:top_n]

# Separate the interaction terms and their importances
interaction_terms, importances = zip(*top_interactions)

# Create a bar plot
plt.figure(figsize=(12, 6))
plt.bar(interaction_terms, importances)
plt.xlabel('Interaction Terms')
plt.ylabel('Importance')
plt.title('Top 15 Interaction Terms in Logistic Regression Model for CIP')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

def plot_interaction_analysis(model_importance_score, X_train, y_train, top_n=15):
    # Extract the feature names from the model_importance_score list and convert them to integers
    top_features = [int(feature) for feature, importance in model_importance_score]

    # Convert X_train to a DataFrame
    X_train_df = pd.DataFrame(X_train)

    # Find the indices of the top features in the original dataset
    top_feature_indices = [X_train_df.columns.get_loc(str(feature)) for feature, importance in model_importance_score]

    # Create a dataset with only the top important features using the feature indices
    X_train_top = X_train_df.iloc[:, top_feature_indices]

    # Create interaction terms
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_interaction = interaction.fit_transform(X_train_top)

    # Train a model with interaction terms
    logreg_interaction = make_pipeline(StandardScaler(), LogisticRegression())
    logreg_interaction.fit(X_train_interaction, y_train)

    # Extract feature importances (coefficients) for the interaction terms
    coef_interaction = logreg_interaction.named_steps['logisticregression'].coef_
    interaction_importances = np.abs(coef_interaction)

    # Analyze the importance of interaction terms
    interaction_terms = interaction.get_feature_names_out(input_features=X_train_top.columns)
    important_interactions = sorted(zip(interaction_terms, interaction_importances[0]), key=lambda x: -x[1])

    # Select the top n interaction terms
    top_interactions = important_interactions[:top_n]

    # Separate the interaction terms and their importances
    interaction_terms, importances = zip(*top_interactions)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(interaction_terms, importances)
    plt.xlabel('Interaction Terms')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Interaction Terms in Logistic Regression Model')
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier


def plot_interaction_analysis(model_importance_score, X_train, y_train, model_type='LogisticRegression', top_n=15):
    # Extract the feature names from the model_importance_score list and convert them to integers
    top_features = [int(feature) for feature, importance in model_importance_score]

    # Convert X_train to a DataFrame
    X_train_df = pd.DataFrame(X_train)

    # Find the indices of the top features in the original dataset
    top_feature_indices = [X_train_df.columns.get_loc(str(feature)) for feature, importance in model_importance_score]

    # Create a dataset with only the top important features using the feature indices
    X_train_top = X_train_df.iloc[:, top_feature_indices]

    # Create interaction terms
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_interaction = interaction.fit_transform(X_train_top)

    # Train a model with interaction terms
    if model_type == 'LogisticRegression':
        clf_interaction = make_pipeline(StandardScaler(), LogisticRegression())
    elif model_type == 'RandomForest':
        clf_interaction = RandomForestClassifier()
    elif model_type == 'SVM':
        clf_interaction = make_pipeline(StandardScaler(), SVC())
    elif model_type == 'XGBoost':
        clf_interaction = XGBClassifier()
    elif model_type == 'GradientBoosting':
        clf_interaction = GradientBoostingClassifier()
    elif model_type == 'LGBM':
        clf_interaction = LGBMClassifier()
    elif model_type == 'CatBoost':
        clf_interaction = CatBoostClassifier(logging_level='Silent')
    elif model_type == 'MLPClassifier':
        clf_interaction = MLPClassifier(max_iter=1000, hidden_layer_sizes=(128,), solver='adam', activation='relu', verbose=False)
    else:
        raise ValueError('Invalid classifier type')

    clf_interaction.fit(X_train_interaction, y_train)

    # Extract feature importances (coefficients) for the interaction terms
    # Extract feature importances (coefficients) for the interaction terms
    if model_type == 'LogisticRegression':
        coef_interaction = list(clf_interaction.named_steps['logisticregression'].coef_[0])
    elif model_type == 'RandomForest':
        coef_interaction = list(clf_interaction.feature_importances_)
    elif model_type == 'SVM':
        coef_interaction = list(clf_interaction.named_steps['svc'].coef_[0])
    elif model_type == 'XGBoost':
        coef_interaction = list(clf_interaction.feature_importances_)
    elif model_type == 'GradientBoosting':
        coef_interaction = list(clf_interaction.feature_importances_)
    elif model_type == 'LGBM':
        coef_interaction = list(clf_interaction.feature_importances_)
    elif model_type == 'CatBoost':
        coef_interaction = list(clf_interaction.get_feature_importance())
    elif model_type == 'MLPClassifier':
        coef_interaction = list(clf_interaction.coefs_[0])


    interaction_importances = np.abs(coef_interaction)

    # Analyze the importance of interaction terms
    interaction_terms = interaction.get_feature_names_out(input_features=X_train_top.columns)
    interaction_importances = interaction_importances.tolist()  # convert array to list
    important_interactions = sorted(zip(interaction_terms, interaction_importances), key=lambda x: -x[1])

    # Select the top n interaction terms
    top_interactions = important_interactions[:top_n]

    # Separate the interaction terms and their importances
    interaction_terms, importances = zip(*top_interactions)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(interaction_terms, importances)
    plt.xlabel('Interaction Terms')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Interaction Terms in {model_type} Model for CIP')
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()



# %%
plot_interaction_analysis(xgb_importance, X_train, y_train, top_n=15, model_type='XGBoost')

# %%
plot_interaction_analysis(lr_importance, X_train, y_train, top_n=15, model_type='LogisticRegression')

# %%
plot_interaction_analysis(rf_importance, X_train, y_train, top_n=15, model_type='RandomForest')

# %%
plot_interaction_analysis(lgb_importance, X_train, y_train, top_n=15, model_type='LGBM')

# %%
plot_interaction_analysis(cb_importance, X_train, y_train, top_n=15, model_type='CatBoost')

# %%
plot_interaction_analysis(gb_importance, X_train, y_train, top_n=15, model_type='GradientBoosting')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def plot_interaction_analysis_MLP(model_importance_score, X_train, y_train, top_n=15):
    # Extract the feature names from the model_importance_score list and convert them to integers
    top_features = [int(feature) for feature, importance in model_importance_score]

    # Convert X_train to a DataFrame
    X_train_df = pd.DataFrame(X_train)

    # Find the indices of the top features in the original dataset
    top_feature_indices = [X_train_df.columns.get_loc(str(feature)) for feature, importance in model_importance_score]

    # Create a dataset with only the top important features using the feature indices
    X_train_top = X_train_df.iloc[:, top_feature_indices]

    # Create interaction terms
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_interaction = interaction.fit_transform(X_train_top)

    # Train a model with interaction terms
    clf_interaction = MLPClassifier(max_iter=1000, hidden_layer_sizes=(128,), solver='adam', activation='relu', verbose=False)
    clf_interaction.fit(X_train_interaction, y_train)

    # Extract feature importances (coefficients) for the interaction terms
    coef_interaction = np.concatenate([coef.flatten() for coef in clf_interaction.coefs_])
    interaction_importances = np.abs(coef_interaction)

    # Analyze the importance of interaction terms
    interaction_terms = interaction.get_feature_names_out(input_features=X_train_top.columns)
    important_interactions = sorted(zip(interaction_terms, interaction_importances), key=lambda x: -x[1])

    # Select the top n interaction terms
    top_interactions = important_interactions[:top_n]

    # Separate the interaction terms and their importances
    interaction_terms, importances = zip(*top_interactions)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(interaction_terms, importances)
    plt.xlabel('Interaction Terms')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Interaction Terms in Feed Forward NN Model for CIP')
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()


# %%
plot_interaction_analysis_MLP(keras_importance, X_train, y_train, top_n=15)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def plot_interaction_analysis_SVM(model_importance_score, X_train, y_train, top_n=15):
    # Extract the feature names from the model_importance_score list and convert them to integers
    top_features = [int(feature) for feature, importance in model_importance_score]

    # Convert X_train to a DataFrame
    X_train_df = pd.DataFrame(X_train)

    # Find the indices of the top features in the original dataset
    top_feature_indices = [X_train_df.columns.get_loc(str(feature)) for feature, importance in model_importance_score]

    # Create a dataset with only the top important features using the feature indices
    X_train_top = X_train_df.iloc[:, top_feature_indices]

    # Create interaction terms
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_interaction = interaction.fit_transform(X_train_top)

    # Train a model with interaction terms
    clf_interaction = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    clf_interaction.fit(X_train_interaction, y_train)

    # Extract feature importances (coefficients) for the interaction terms
    coef_interaction = clf_interaction.named_steps['svc'].coef_
    interaction_importances = np.abs(coef_interaction)

    # Analyze the importance of interaction terms
    interaction_terms = interaction.get_feature_names_out(input_features=X_train_top.columns)
    important_interactions = sorted(zip(interaction_terms, interaction_importances[0]), key=lambda x: -x[1])

    # Select the top n interaction terms
    top_interactions = important_interactions[:top_n]

    # Separate the interaction terms and their importances
    interaction_terms, importances = zip(*top_interactions)

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(interaction_terms, importances)
    plt.xlabel('Interaction Terms')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Interaction Terms in SVM Model for CIP')
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.show()


# %%
plot_interaction_analysis_SVM(svm_importance, X_train, y_train, top_n=15)

# %%


# %%


# %% [markdown]
# ## Evaluation on Uganda Data

# %%
from sklearn.metrics import roc_auc_score

# Initialize a dictionary to store evaluation metrics for each model
Uganda_metrics = {}

# Define a function to calculate the evaluation metrics
def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }

# Evaluate each model on the independent dataset
for model_name, model in classifiers.items():
    y_pred = model.predict(Uganda_X)
    y_pred_proba = model.predict_proba(Uganda_X)[:, 1]
    Uganda_metrics[model_name] = calculate_metrics(Uganda_y, y_pred, y_pred_proba)




# %%
#Plot Uganda metrics
plot_metrics(Uganda_metrics, "Uganda CIP")

# %%
Uganda_df = pd.DataFrame(Uganda_metrics).transpose().round(2)
print(Uganda_df)

# %%
Uganda_df.to_csv('Uganda_metrics_balanced_CIP.csv')

# %%


# %%


# %%
results

# %%
Uganda

# %%
#Plot Uganda ROC curves
plot_roc_curves(Uganda_metrics, Uganda_X, Uganda_y, "Uganda CIP")

# %%
#Perfrom k-fold cross validation for Uganda CIP
Uganda_auc_scores_LR = cross_val_score(classifiers['Logistic Regression'], Uganda_X, Uganda_y)

# %%


# %%


# %%


# %%
Uganda_auc_scores_RF = cross_val_score(classifiers['Random Forest'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_GB = cross_val_score(classifiers['Gradient Boosting'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_XGB = cross_val_score(classifiers['XGBoost'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_LGBM = cross_val_score(classifiers['LightGBM'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_CAT = cross_val_score(classifiers['CatBoost'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_NN = cross_val_score(classifiers['Feed-Forward NN (Keras)'], Uganda_X, Uganda_y)


# %%
Uganda_auc_scores_SVM = cross_val_score(classifiers['SVM'], Uganda_X, Uganda_y)

# %%
#Store all the individual scores in Ugada_auc_scores together with the model names
Uganda_auc_scores = {'Logistic Regression': Uganda_auc_scores_LR, 'Random Forest': Uganda_auc_scores_RF, 'Gradient Boosting': Uganda_auc_scores_GB, 'XGBoost': Uganda_auc_scores_XGB, 'LightGBM': Uganda_auc_scores_LGBM, 'CatBoost': Uganda_auc_scores_CAT, 'Feed-Forward NN (Keras)': Uganda_auc_scores_NN, 'SVM': Uganda_auc_scores_SVM}
Uganda_auc_scores
                                    

# %%
#Violin plot for Uganda CIP
plot_auc_boxplot_violinplot(Uganda_auc_scores, "Uganda CIP")

# %%
print_mean_auc_scores(Uganda_auc_scores)


# %%
auc_multiple_comparison_test(Uganda_auc_scores)

# %%


# %%


# %%


# %%


# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrices(label, y_true, classifiers, X_data, model_names, figsize=(22, 12)):
    n_models = len(classifiers)
    fig, axes = plt.subplots(nrows=2, ncols=(n_models + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for ax, model_name, model in zip(axes, model_names, classifiers.values()):
        y_pred = model.predict(X_data)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{model_name} Confusion Matrix for {label} Data")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        #Print cm values for each model
        print(f"{model_name} Confusion Matrix")
        print(cm)
    
    plt.tight_layout()
    plt.show()

plot_confusion_matrices("imbalanced CIP", y, classifiers, X, model_names)


# %%
plot_confusion_matrices("Uganda CIP", Uganda_y, classifiers, Uganda_X, model_names)


# %%
def plot_class_distribution(y, title='Class Distribution', figsize=(8, 6)):
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')
    plt.show()

plot_class_distribution(y, 'England CIP Dataset Class Distribution')


# %%
def plot_class_distribution_Uganda(Uganda_y, title='Class Distribution', figsize=(8, 6)):
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=Uganda_y)
    plt.title(title)
    plt.xlabel("Class")
    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 10), 
                    textcoords = 'offset points')
    plt.ylabel("Count")

    plt.show()

plot_class_distribution_Uganda(Uganda_y, 'Uganda CIP Dataset Class Distribution')

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
  for name, model in models:  
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(name)
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('Matthews Correlation Coefficient')
    print(matthews_corrcoef(y_test, y_pred))
    print('')
    #Evaluate on test and train
    print('Train Accuracy')
    print(pipeline.score(X_train, y_train))
    print('Test Accuracy')
    print(pipeline.score(X_test, y_test))
    print('')


    

# %%
#Use Voting Classifier to combine the models
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


evc = VotingClassifier(estimators=models, voting='hard')
evc.fit(X_train, y_train)

y_pred = evc.predict(X_test)
print(classification_report(y_test, y_pred))



# %%
import tensorflow as tf
import keras as K

print(tf.__version__)
print(K.__version__)


# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Convert y to integer labels using LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y)

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the architecture of the deep neural network
model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Train the model on the training data, using the validation data to monitor performance
history = model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val), verbose=2)

# Evaluate the performance of the trained model on the training data
y_train_pred = model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)
train_accuracy = accuracy_score(y_train, y_train_pred_classes)
print('Train accuracy:', train_accuracy)

# Evaluate the performance of the trained model on the testing data
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
test_accuracy = accuracy_score(y_test, y_test_pred_classes)
print('Test accuracy:', test_accuracy)

# Print classification report
target_names = ['Susceptible', 'Resistant']
print(classification_report(y_test, y_test_pred_classes, target_names=target_names))


# %%


# %%


# %%


# %%


# %% [markdown]
# ### Possibilities of why the model is not perfoming well on the 0 class
# - Class imbalance: If one outcome has very few samples in the training set, the model may not be able to learn enough about that outcome to make accurate predictions. This can result in low precision, recall, and F1 scores for that outcome.
# 
# - Noisy or insufficient data: If the data for one outcome is particularly noisy or has insufficient information to distinguish it from other outcomes, the model may struggle to learn how to predict that outcome accurately.

# %%


# %%


# %%


# %%


# %% [markdown]
# ### Hyperparameters from small dataset
# Best Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 97}
# Best Accuracy Score: 0.7964847363552268
# 
# These hyperparameters were found by the RandomizedSearchCV to give the best accuracy score on the training data. We can use these hyperparameters to train a Random Forest Classifier on the full training set and evaluate its performance on the test set to see how well it generalizes to new, unseen data.

# %%


# %%
#Create a new model object with the best hyperparameters
from sklearn.metrics import accuracy_score


best_model = RandomForestClassifier(**random_search.best_params_, random_state=42)

#Fit the model to the training data
best_model.fit(X_train, y_train)

#Predict the labels of the test set
y_pred = best_model.predict(X_test)

#Print the accuracy score
print('Accuracy Score:', accuracy_score(y_test, y_pred))

#Print the confusion matrix
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

#Print the classification report
print('Classification Report:', classification_report(y_test, y_pred))




# %%
# Fit the best model to the entire training set
##best_model.fit(X_train, y_train)
#Predict the response for test dataset
#y_pred = best_model.predict(X_test)


# %%
#Identify the most important features
feature_importances = random_search.best_estimator_.feature_importances_
most_important_features = data.columns[1:][feature_importances.argsort()[::-1][:30]]
print(f'Most important features: {most_important_features}')


# %%
#Plot the most important features and their importance in P value
plt.figure(figsize=(10, 5))
plt.barh(most_important_features, feature_importances[feature_importances.argsort()[::-1][:30]])
plt.title('Most important features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# %% [markdown]
# ### Compute the combinatation of features 
# We want to view how a combination of features perform 
#   
# 
# 

# %%
from itertools import combinations
from sklearn.inspection import permutation_importance

# Define a function to get all possible feature combinations
def get_feature_combinations(features):
    all_combinations = []
    for k in range(1, len(features) + 1):
        for subset in combinations(features, k):
            all_combinations.append(list(subset))
    return all_combinations

# %%
import random
random_features = random.sample(most_important_features.tolist(), k=10)

# %%
feature_combinations = get_feature_combinations(random_features)

# %%
from itertools import combinations
from sklearn.inspection import permutation_importance

# Define a function to get all possible feature combinations
def get_feature_combinations(features):
    all_combinations = []
    for k in range(1, len(features) + 1):
        for subset in combinations(features, k):
            all_combinations.append(list(subset))
    return all_combinations

# Get all possible feature combinations
feature_combinations = get_feature_combinations(X.columns.tolist())

# Calculate feature importances for each combination
importances = []
for features in feature_combinations:
    rfc = RandomForestClassifier(
        n_estimators=97,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    rfc.fit(X_train[features], y_train)
    result = permutation_importance(rfc, X_test[features], y_test, n_repeats=5, random_state=42)
    importances.append((features, result.importances_mean))

# Sort the feature importances by descending order
importances = sorted(importances, key=lambda x: sum(x[1]), reverse=True)

# Print the top 10 feature combinations with their importances
for features, importance in importances[:10]:
    print(f'{features}: {sum(importance)}')


# %%


# %%


# %%


# %%


# %%
Independent_X = Uganda_X
Independent_y = Uganda_y

# %%
#Validate the best model on an independent dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

independent_y_pred = random_search.predict(Independent_X)
independent_accuracy = accuracy_score(Independent_y, independent_y_pred)
independent_precision = precision_score(Independent_y, independent_y_pred, average='weighted')
independent_recall = recall_score(Independent_y, independent_y_pred, average='weighted')
independent_f1 = f1_score(Independent_y, independent_y_pred, average='weighted')

print(f'Accuracy: {independent_accuracy}, Precision: {independent_precision}, Recall: {independent_recall}, F1-score: {independent_f1}')



# %%
#Classification report for the independent dataset
print(classification_report(Independent_y, independent_y_pred))


# %%


# %%


# %%


# %%
#calculate the auc score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

# calculate the fpr and tpr for all thresholds of the classification
probs = random_search.predict_proba(X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




# %%
#calculate the auc score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

# calculate the fpr and tpr for all thresholds of the classification
probs = random_search.predict_proba(Independent_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(Independent_y, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ### Automation of the workflow for all the models altogether

# %%
#A loop to find the best hyperparameters for all the models and their classication report


# %%
data.head(3)

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix 
from sklearn.metrics import matthews_corrcoef,auc, roc_curve,plot_roc_curve, plot_precision_recall_curve,classification_report, confusion_matrix,average_precision_score, precision_recall_curve

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np

# Define the models to evaluate
models = [
    ('Logistic Regression', LogisticRegression(max_iter=10000), {'C': [0.1, 1, 10]}),
    ('Random Forest', RandomForestClassifier(), {'n_estimators': randint(10, 100), 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    #('Gradient Boosting', GradientBoostingClassifier(), {'n_estimators': randint(10, 100), 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'learning_rate': [0.001, 0.01, 0.1, 1]}),
    #('Support Vector Machine', SVC(), {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'sigmoid']}),
]

#Initialise empty lists for all metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
model_feature_importances = []
MCC_scores = []
fpr = []
tpr = []
auc_roc_scores = []
confusion_matrices = []
classification_reports = []





for name, model, hyperparameters in models:
    pipeline = make_pipeline(StandardScaler(), model)

    # Perform k-fold cross validation on the model
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f'{name}: {np.mean(scores)} +/- {np.std(scores)}')

    # Define the randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=hyperparameters,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Fit the randomized search to the data
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters and accuracy score
    print('Best Hyperparameters:', random_search.best_params_)
    print('Best Accuracy Score:', random_search.best_score_)

    # Evaluate the model on the test set using the best hyperparameters
    model = model.set_params(**random_search.best_params_)
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate the metrics and store them in the lists
    precision = precision_score(y_test, y_pred, average='weighted')
    precision_scores.append((name, precision))
    recall = recall_score(y_test, y_pred, average='weighted')
    recall_scores.append((name, recall))
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append((name, f1))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append((name, accuracy))
    confusion_matrices.append((name, confusion_matrix(y_test, y_pred)))
    classification_reports.append((name, classification_report(y_test, y_pred)))
    MCC = matthews_corrcoef(y_test, y_pred)
    MCC_scores.append((name, MCC))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_roc = auc(fpr, tpr)
    auc_roc_scores.append((name, auc_roc))

    # Identify the most important 10 features for each model
    if name == 'Logistic Regression':
        feature_importances = random_search.best_estimator_.coef_[0]
        
        most_important_features = data.columns[1:][np.abs(feature_importances).argsort()[::-1][:10]]
        model_feature_importances.append((name, most_important_features))
    elif name == 'Random Forest':
        feature_importances = random_search.best_estimator_.feature_importances_
        most_important_features = data.columns[1:][np.abs(feature_importances).argsort()[::-1][:10]]
        model_feature_importances.append((name, most_important_features))
    elif name == 'Gradient Boosting':
        feature_importances = random_search.best_estimator_.feature_importances_
        most_important_features = data.columns[1:][np.abs(feature_importances).argsort()[::-1][:10]]
        model_feature_importances.append((name, most_important_features))
    elif name == 'Support Vector Machine':
        feature_importances = random_search.best_estimator_.coef_[0]
        most_important_features = data.columns[1:][np.abs(feature_importances).argsort()[::-1][:10]]
        model_feature_importances.append((name, most_important_features))
    else:
        raise ValueError("Estimator not supported")
    
    
    # Print the classification report
    print(f'Classification Report for {name}:')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))
    print('F1:', f1_score(y_test, y_pred, average='weighted'))

# %%
auc_roc_scores

# %%
#Plot the ROC curve
from sklearn.metrics import plot_confusion_matrix


for name, auc_roc in auc_roc_scores:
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()





# %%
#Plot features importances for each model
model_feature_importances



