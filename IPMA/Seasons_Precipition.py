import os

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

month_abbreviations = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec'
}

month_season = {
    'Jan': "winter",
    'Feb': "winter",
    'Mar': "spring",
    'Apr': "spring",
    'May': "spring",
    'Jun': "summer",
    'Jul': "summer",
    'Aug': "summer",
    'Sep': "fall",
    'Oct': "fall",
    'Nov': "fall",
    'Dec': "winter"
}

decode_season = {
    3 : "winter",
    1 : "spring",
    2 : "summer",
    0 : "fall",

}

encode_season = {
    "winter": 3,
    "spring": 1,
    "summer": 2,
    "fall": 0,
}

cdict = {
    0: 'rosybrown',
    1: 'springgreen',
    2: 'goldenrod',
    3: 'deepskyblue'
}

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(50),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1),
    MLPClassifier(learning_rate='adaptive',max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

#PRECIPITION
prec = pd.read_csv("prec-Lisboa_Geofísico-hom.csv")
#drop non months columns
prec = prec.drop(["Winter","Spring","Summer","Autumn","Annual"],axis=1)

id_vars = ['year']
value_vars = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
prec_vertical = prec.melt(id_vars=id_vars, value_vars=value_vars, var_name='month', value_name='prec')


#TEMPERATURE
temp = pd.read_csv("temp-Lisboa_Geofísico-hom.csv")
temp['date'] = pd.to_datetime(temp['date'], format='%m/%Y')
temp['year'] = temp['date'].dt.year
temp['month'] = temp['date'].dt.month.map(month_abbreviations)
temp = temp.drop(columns=['date'])
columns = ['year', 'month'] + [col for col in temp.columns if col not in ['year', 'month']]
temp = temp.reindex(columns=columns)

#PRESSURE
press = pd.read_csv("press-Lisboa_Geofísico-hom.csv")

press['date'] = pd.to_datetime(press['date'], format='%m/%Y')
press['year'] = press['date'].dt.year
press['month'] = press['date'].dt.month.map(month_abbreviations)
press = press.drop(columns=['date'])

columns = ['year', 'month'] + [col for col in press.columns if col not in ['year', 'month']]
press = press.reindex(columns=columns)

m1 = pd.merge(prec_vertical,temp,on=["year","month"])
df_merged = pd.merge(m1, press, on=['year', 'month'])

df_merged["season"] = df_merged["month"].map(month_season)

data = df_merged.drop(["year","month","tmin","tmax","press"],axis=1).dropna()


X_data = data.drop(["season"],axis=1)
y_data = data["season"].map(encode_season)
#pca = PCA(n_components=2)
#pca.fit(X_data)
X_pca = X_data #pca.transform(X_data)


X_train, X_test, y_train, y_test = train_test_split(X_pca, y_data, test_size=0.1)
x_min, x_max = X_pca["prec"].min() - 0.5, X_pca["prec"].max() + 0.5
y_min, y_max = X_pca["tmed"].min() - 0.5, X_pca["tmed"].max() + 0.5
i= 0
fig, ax = plt.subplots(2, 5, figsize=(30, 10))
for name, clf in zip(names,classifiers):
    print(name)
    x = i//5
    y = i%5
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X_pca, response_method="predict",
        xlabel="Precipitation", ylabel="Mean Temperature", ax=ax[x,y],
        alpha=0.5, cmap = matplotlib.colors.ListedColormap(['rosybrown','springgreen','goldenrod','deepskyblue'])
    )
    X_test = np.array(X_test)
    for g in np.unique(y_test):
        ix = np.where(y_test == g)
        ax[x, y].scatter(X_test[ix, 0], X_test[ix, 1], c=cdict[g], label=decode_season[g], edgecolor="k")
    ax[x, y].set_xlim(x_min, x_max)
    ax[x, y].set_ylim(y_min, y_max)
    ax[x, y].set_xticks(())
    ax[x, y].set_yticks(())
    ax[x, y].set_title(name)
    ax[x, y].text(
        x_max - 0.3,
        y_min + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    ax[x, y].legend()
    i += 1

plt.tight_layout()
plt.savefig('results4.png')
plt.show()
