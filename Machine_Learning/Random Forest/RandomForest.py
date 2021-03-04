def RandomForest(data):
    import pandas as pd
    import numpy as np
    dataset = pd.read_csv(data)
    X = dataset[['math score','reading score','writing score']]
    y = dataset['admitted']
    y=y.astype(int)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier

    regressor = RandomForestClassifier(n_estimators=200)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)


    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))
    #use the feature importance variable to see feature importance scores.
    feature_names=[["math score","reading score","writing score"]]
    import pandas as pd
    feature_imp = pd.Series(regressor.feature_importances_,index=feature_names).sort_values(ascending=False)
    print(feature_imp)
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()
    from sklearn.metrics import classification_report, confusion_matrix
    print("matrice du confusion :\n",confusion_matrix(y_test, y_pred))
    from sklearn import metrics
    print("Accuracy du modèle :",metrics.accuracy_score(y_test, y_pred))
    print("Precision du modéle :",metrics.precision_score(y_test, y_pred))
    print("Recall du modéle:",metrics.recall_score(y_test, y_pred))
    print("rapport du classification :\n",classification_report(y_test, y_pred))
        
if __name__=="__main__":
    RandomForest("myData.csv")

