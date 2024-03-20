import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
# data = pd.read_csv("IRIS.csv")
iris = load_iris(as_frame=True)
# Data preprocessing
# 1. Check for missing values
# https://www.geeksforgeeks.org/working-with-missing-data-in-pandas/#:~:text=In%20order%20to%20check%20missing,are%20True%20for%20NaN%20values
missing_val = pd.isnull(iris.data['sepal length (cm)'])
# We don't have any missing values, so we can proceed.

# Split the data into features and outcome
X = iris.data
y = iris.target
# X_df = pd.DataFrame(iris.data)
# y_df = pd.DataFrame(iris.target)
# X_df['target'] = iris.target
# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Printing metrics of the model

# I prefer average=micro, because it takes into account contributions of all class equally. For example, model can missclasify
# one class, but it won't influence the overall statistics significantly.
# Accuracy = TP + TN / TP + TN + FP + FN

print("Accuracy score of the model (Decision tree):", metrics.accuracy_score(y_test, y_pred))
# Precision = TP / TP + FP
print("Precision score of the model (Decision tree):", metrics.precision_score(y_test, y_pred, average='micro'))
# Recall = TP / TP + FN
print("Recall score of the model (Decision tree):", metrics.recall_score(y_test, y_pred, average='micro'))
# F1 - score = 2 * Precision * Recall /  Precision + Recall
print("f1 - score of the model (Decision tree):", metrics.f1_score(y_test, y_pred, average='micro'))

#F1 Score is the best measure for this task.
# 1) we follow a balance between Precision and Recall (they have the same importance for this task (in my opinion)
# 2) Uneven class distribution (large number of Actual Negatives). There are 3 classes, 50 elements for each, therefore
# we have 50 true values and 100 negatives.
# sns.set_style("whitegrid")
# sns.FacetGrid(X_df, hue="target",
#               height = 6).map(plt.scatter,
#                               'sepal length (cm)',
#                               'sepal width (cm)').add_legend()
# plt.show()
pred_df = pd.DataFrame(iris.data)
X_test['target'] = y_test
X_test['predicted'] = y_pred
sns.set_style("whitegrid")
sns.FacetGrid(X_test, hue="target",
              height = 6).map(plt.scatter,
                              'sepal length (cm)',
                              'sepal width (cm)').add_legend()
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(X_test, hue="predicted",
              height = 6).map(plt.scatter,
                              'sepal length (cm)',
                              'sepal width (cm)').add_legend()
plt.show()


# Create a scatter plot for y_test
# i = 0
# fig, axes = plt.subplots(1, len(X.columns), figsize=(30, 5), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
#
# for s in X.columns:
#     ax = axes[i]
#     ax.set_xlabel(s)
#     ax.set_title(s + ' and classification')
#
#     if i == 0:
#         ax.plot(X_test[s], y_pred, 'go', label='Prediction by a model')
#         ax.plot(X_test[s], y_test, 'ro', label='actual data')
#         ax.legend()
#     else:
#         ax.plot(X_test[s], y_pred, 'go')
#         ax.plot(X_test[s], y_test, 'ro')
#     i += 1
#     print(X_test[s].size, y_test.size)
# plt.show()
