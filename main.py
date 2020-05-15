from scripts.knn import *


DATA_PATH = r'C:\Users\MSI\PycharmProjects\KNN Classifier\data\iris.data'
NAMES = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
SPLIT = 0.7


df = load_data(DATA_PATH, names=COLUMNS)
df = df.sample(frac=1)
labels = df['species']
sl = df['sepal_length']
sw = df['sepal_width']
pl = df['petal_length']
pw = df['petal_width']
features = list(zip(sl, sw, pl, pw))

labels = encode_labels(labels)
n = int(len(features)*SPLIT)
x_train, x_test = features[:n], features[n:]
y_train, y_test = labels[:n], labels[n:]

knn = KNearestNeighbors()
knn.fit(x_train, y_train)
y_hat = knn.predict(x_test, neighbors=5, metrics='minkowski')
res = knn.evaluate(y_test, y_hat)
print('acc: %.2f' % res)
