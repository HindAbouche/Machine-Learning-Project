def RégressionLogistiqueMulticlasse(data):
    #importation des libraries 
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    dataset=pd.read_csv(data)
    X = dataset[['math score', 'reading score','writing score']]
    y = dataset['parental level of education']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    # Mise à l'échelle des fonctionnalités
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Ajuster le classificateur à train data
    # Créez notreclassificateur 
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
    classifier.fit(X_train, y_train)

    # Prédire les résultats de test data
    y_pred = classifier.predict(X_test)

    # Créer la matrice de confusion
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    import seaborn as sn
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))

if __name__=='__main__':
    RégressionLogistiqueMulticlasse("myData.csv")

