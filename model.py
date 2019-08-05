import pandas as pd
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('data_train_1.csv')
y_data = data['stroke_in_2018']
data.drop(columns=['stroke_in_2018'],inplace=True)
print(data.shape)
print(y_data.shape)

train_x, test_x, train_y, test_y = model_selection.train_test_split(data, y_data,test_size=0.3)

print(train_x.shape)
print(train_y.shape)
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    #     gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    #     svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    #     discriminant_analysis.LinearDiscriminantAnalysis(),
    #     discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()
    # lightboost
]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3,
                                        train_size = .6, random_state = 0 )
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean',
               'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
MLA_predict = deepcopy(train_y)
print(train_x.shape)
print(train_y.shape)
row_index = 0
for alg in MLA:
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    print(MLA_name)
    print(alg.get_params())
    cv_results = model_selection.cross_validate(alg, train_x, train_y, cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA '] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    alg.fit(train_x, train_y)
    MLA_predict[MLA_name] = alg.predict(train_x)
    row_index+=1

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

print(MLA_compare)


