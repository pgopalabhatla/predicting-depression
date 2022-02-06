# Load packages we need
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
import sklearn
import pickle


from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})


# Let's check our software versions
print('### Python version: ' + __import__('sys').version)
print('### NumPy version: ' + np.__version__)
print('### Scikit-learn version: ' + sklearn.__version__)
print('### Pickle version: ' + pickle.format_version)
print('------------')


# Read the data
df = pd.read_csv('data.csv', sep='\t')


# Clean the data
for col in df.columns:
    if 'E' in col:
        df.drop(columns=col, inplace=True)
    elif 'TIPI' in col:
        if '4' not in col:
            df.drop(columns=col, inplace=True)
    elif 'I' in col:
        df.drop(columns=col, inplace=True)
    elif 'Q' not in col:
        df.drop(columns=col, inplace=True)


# -------- LinearSVC -------- #


# all_xy = np.asarray(df, dtype='float64') #matrix
# assert all_xy.shape[1] == 43
#
#
# # ## Feature Engineering
#
#
#
# x_idx = range(0, 42)
# all_x = all_xy[:,x_idx]
# all_y = all_xy[:,42]
#
#
#
#
# print('features: ', all_x)
# print('label: ', all_y)
#
#
#
#
# for x in df.columns:
#     y = df[x].value_counts()
#     print(y)
#
#
# # ## Train Split Validation
#
#
#
# seed = 42
# np.random.seed(seed)
# prop_vec = [14, 3, 3]
# train_x, train_y, test_x, test_y, val_x, val_y = utils.train_test_val_split(all_x, all_y, prop_vec,
#                                                                                            shuffle=True, seed=seed-1)
#
#
#
#
# train_xy = np.hstack((train_x, train_y.reshape(-1,1)))
# pairwise_corr = np.corrcoef(train_xy, rowvar=False)     #rowvar shows each col is var
# plots.heatmap(pairwise_corr, df.columns, df.columns, rot=90, fsz=(14, 14), colorbar=True)
#
#
# # ## SVM
#
#
#
# # not using this one since accuracy isn't high enough
#
# from sklearn.svm import SVC, LinearSVC
# from sklearn.base import clone
#
# svm = SVC(kernel='linear', random_state=seed)
#
# svm.fit(train_x, train_y)
#
#
#
#
# def model_accuracy(model, x, true_y):
#     pred = model.predict(x)
#     return np.sum(pred == true_y) / true_y.shape[0]
#
# name = 'svm'
# train_acc = model_accuracy(svm, train_x, train_y)
# val_acc = model_accuracy(svm, val_x, val_y)
#
# print('[{}] Train accuracy: {:.2f}%, Val accuracy: {:.2f}%'.format(name, train_acc*100, val_acc*100))

# Accuracy : 42%

# --------  -------- #

# -------- RainForestClassifier -------- #

## Restructure the label to categorical form
## Giving the self-reported values 0-3 as label 0 (no anxiety) and 4-7 as label 1 (anxiety)
d_map = {0: 0,
         1: 0,
         2: 0,
         3: 0,
         4: 1,
         5: 1,
         6: 1,
         7: 1}
df['TIPI4'] = df['TIPI4'].map(d_map)

## Dividing the features and label
X = df.drop(columns='TIPI4')
y = df['TIPI4']

## Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

## Loading the model
random_forest_model = RandomForestClassifier(random_state=10)

## Fitting the model to the training set
random_forest_model.fit(X_train, y_train.ravel())

## Predicting the model on the test set
predict_train_data = random_forest_model.predict(X_test)

# name = 'random forest classifier'

## Printing the accuracy
# score = metrics.accuracy_score(y_test, predict_train_data)
# score = score * 100
#
# print("[{}] Accuracy: {:.2f}%".format(name, score))

# Accuracy: 84.62%

# --------  -------- #

# -------- Saving the model -------- #

pickle.dump(random_forest_model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

# --------  -------- #

print("Ran without any errors!")
