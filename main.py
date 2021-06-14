import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, plot_confusion_matrix

# data reading
df = pd.read_csv("database.txt", delimiter=",", header=None)

# change last 2 column name as a latitude and longitude
column_names = ["Feature " + str(x) for x in range(68)]
column_names.append('latitude')
column_names.append('longitude')
df.columns = column_names

df_temp = df.copy()
print(df.head())


longitude = np.array(df[['longitude']])
# df.drop(df.columns[[69]], inplace = True, axis = 1)
latitude = np.array(df[['latitude']])
# df.drop(df.columns[[68]], inplace = True, axis=1)

# convert pandas df to geopandas df
gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(longitude, latitude))
print(gdf.head())


world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = world.plot(color='white', edgecolor='black', markersize=20, label="Countries")
# We can now plot our ``GeoDataFrame``.
ax.set_title("Country Labels")
gdf.plot(ax=ax, color='red')
plt.show()

# convert latitude and longitude values to country names
# now target values are categorical values
label_text = {(-15.75, -47.95): 'Brasil', (14.91, -23.51): 'Cape Verde', (12.65, -8.0): 'Mali',
              (9.03, 38.74): 'Ethiopia', (34.03, -6.85): 'Morocco', (14.66, -17.41): 'Senegal',
              (52.5, -0.12): 'England', (41.26, 69.21): 'Uzbekistan', (41.9, 12.48): 'Italy',
              (28.61, 77.2): 'India', (33.66, 73.16): 'Pakistan', (54.68, 25.31): 'Lithunia',
              (44.41, 26.1): 'Romania', (36.7, 3.21): 'Algeria', (39.91, 32.83): 'Turkey',
              (19.75, 96.1): 'Myanmar', (13.75, 100.48): 'Thailand', (39.91, 116.38): 'China',
              (23.76, 121.0): 'Taiwan', (-6.17, 106.82): 'Indonesia', (17.98, -76.8): 'Jamaica',
              (35.68, 51.41): 'Iran', (30.03, 31.21): 'Egypt', (42.86, 74.6): 'Kyrgyzstan',
              (-1.26, 36.8): 'Kenya', (17.25, -88.76): 'Belize', (38.0, 23.71): 'Greece',
              (-35.3, 149.12): 'Australia', (35.7, 139.71): 'Japan', (-6.17, 35.74): 'Tanzania',
              (41.71, 44.78): 'Georgia', (11.55, 104.91): 'Cambodia', (41.33, 19.8): 'Albania'
              }

df_labeled = df_temp.copy()
df_labeled['Country'] = list(zip(df_labeled['latitude'], df_labeled['longitude']))
df_labeled['Country'] = df_labeled['Country'].map(label_text)
df_labeled.drop(labels=['latitude', 'longitude'], axis=1, inplace=True)

# showing converted df
df_labeled.Country.value_counts().plot(kind='bar', title="Number of tracks per Country", colormap="gist_gray")
plt.show()

plt.rcParams.update({'font.size': 6})
# df_labeled.drop([68],axis=1)
X = df_labeled.iloc[:, :-1]
y = df_labeled.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=5)


print("******************************************************")
# RANDOM FOREST CLASSIFIER
# n_estimator = 100, 500, 1000 - best accuracy in 1000
# min_samples_leaf = 1, 5 , 10 - best accuracy in 1
rfc = RandomForestClassifier(n_estimators=1000, random_state=1, min_samples_leaf=1)
rfc.fit(X_train, y_train.ravel())
predicted = rfc.predict(X_test)

print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted, zero_division=1))
print(accuracy_score(y_test, predicted))

disp = plot_confusion_matrix(rfc, X_test, y_test, xticks_rotation='vertical')
plt.title("Random Forest Classifier Confusion Matrix")
plt.show()

print("******************************************************")
# LOGISTIC REGRESSION CLASSIFIER
# max_iter = 100, 200, 500, 1000 - best accuracy >500
# multi_class = multinomial, ovr - best accuracy in multinomial
logreg = LogisticRegression(max_iter=500, random_state=1, multi_class='multinomial')
logreg.fit(X_train, y_train.ravel())
predicted = logreg.predict(X_test)

print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted, zero_division=1))
print(accuracy_score(y_test, predicted))

disp = plot_confusion_matrix(logreg, X_test, y_test.ravel(), xticks_rotation='vertical')
plt.title("Logistic Regression Classifier Confusion Matrix")
plt.show()

print("******************************************************")
# SUPPORT VECTOR MACHINE
# kernel = linear, poly, rbf, sigmoid - best accuracy in linear
# C = 0.01, 0.1, 1 - best accuracy in 0.1
linear = svm.SVC(kernel='linear', gamma='scale', C=.1)

linear.fit(X_train, y_train.ravel())
predicted = linear.predict(X_test)
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted, zero_division=1))
print(accuracy_score(y_test, predicted))

disp = plot_confusion_matrix(linear, X_test, y_test.ravel(), xticks_rotation='vertical')
plt.title("Support Vector Machine-Linear Kernel Confusion Matrix")
plt.show()

print("******************************************************")


