#%% importowanie bibliotek
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%% iport danych 

train_data_raw = pd.read_csv(r"input\train.csv")

test_data_raw = pd.read_csv(r"input\test.csv")

y_test_raw = pd.read_csv(r'input\gender_submission.csv')

#%% analiza dandych wejsciowych

#print(train_data.isnull())

print (train_data_raw.info())
print (test_data_raw.info())

#%%
#missing data
total = train_data_raw.isnull().sum().sort_values(ascending=False)
percent = (train_data_raw.isnull().sum()/train_data_raw.isnull().count()).sort_values(ascending=False)*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
print(train_data_raw.isnull().sum().max())

#%% nieistatne wartosci: name, ticket, cabin 
train_data = train_data_raw.drop(["Ticket","Cabin","Name"], axis=1)
test_data = test_data_raw.drop(["Ticket","Cabin","Name"], axis=1)

train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
test_data['Age'].fillna(test_data['Age'].median(), inplace = True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace = True)


train_data['Fare'].fillna(train_data['Fare'].median(), inplace = True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace = True)


print (train_data.info())
print (test_data.info())

#%%String to int 
# Sex: F-1, M-0 , Embarked S-1,C-2,Q-3
train_data['Sex'] = train_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_data['Sex'] = test_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#%%
train_data['Embarked'] = train_data['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
test_data['Embarked'] = test_data['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
#%% 
#correlation matrix
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

#%% klasifikatory
'''
Regresja liniowa (LinearRegression)
Drzewa decyzyjne (DecisionTree)
Lasy drzew decyzyjnych (RandomForest)
Supported Vector Machines (SVM)

'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

reg_clf = LinearRegression()
dt_clf = DecisionTreeClassifier(max_depth=3,random_state=2)
svc_clf = SVC()
rf_clf = RandomForestClassifier(n_estimators=100)


X_train =  train_data.drop('Survived', axis =1)
y_train = train_data['Survived']

X_test = test_data
y_test = y_test_raw['Survived']


#%%



classifiers = [reg_clf, dt_clf, svc_clf, rf_clf]

for clf in classifiers:
  print(f'----{clf}---------')
  print("fitting - training...")
  clf.fit(X_train,y_train)

  #print("training on whole dataset...")
  #clf.fit(X, y)

  print("predicting...")
  y_pred = clf.predict(X_test)

  # wypisujemy wartości dla pierwsyzch 10 predykcji

 # print("true values ", y_train[:10])
  #print("predicted   ", y_pred[:10])

  print("scoring...")

  clf_score = clf.score(X_train,y_train)
  print("Train score = ", clf_score)
  clf_score = clf.score(X_test,y_test)
  print("Test score = ", clf_score)

  #clf_score = clf.score(X,y)
  #print("whole set score = ", clf_score)
  


            