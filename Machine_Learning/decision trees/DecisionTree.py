# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
data = pd.read_csv("myData.csv")

X = data[['math score','reading score','writing score']] 
y = data['admitted'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=100) 


clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy score on train data(using default criterionas gini) ',
      accuracy_score(y_true=y_train,y_pred=clf.predict(X_train)))
print('Accuracy score on test data(using default criterionas gini) ',
      accuracy_score(y_true=y_test,y_pred=y_pred))
print(' an other criterien ')

clf2 = DecisionTreeClassifier(criterion='entropy',min_samples_split=50)


clf2 = clf2.fit(X_train,y_train)

y_pred = clf2.predict(X_test)
print('Accuracy score on train data(using entropy ',accuracy_score(y_true=y_train,y_pred=clf2.predict(X_train)))
print('Accuracy score on test data(using entropy) ',accuracy_score(y_true=y_test,y_pred=y_pred))



from sklearn.metrics import classification_report, confusion_matrix
print("matrice du confusion :\n",confusion_matrix(y_test, y_pred))
from sklearn import metrics
print("Accuracy du modèle :",metrics.accuracy_score(y_test, y_pred))
print("Precision du modéle :",metrics.precision_score(y_test, y_pred))
print("Recall du modéle:",metrics.recall_score(y_test, y_pred))
print("rapport du classification :\n",classification_report(y_test, y_pred))

feature_names=['math score','reading score','writing score']
class_name=['admitted']

print('----------------------visualize the  tree----------------')
from sklearn.tree import  export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,
feature_names=feature_names,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
#enregistrer l'image qui contient le TREE
graph.write_png("tree.png")
#enregistrer le PDF qui contient le TREE

        

"""
Image(graph.create_png())
graph.write_pdf("tree.pdf")"""









