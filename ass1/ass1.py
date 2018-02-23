import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, pairwise
import itertools

train_in = np.genfromtxt('./data/train_in.csv', delimiter=',')
train_out = np.genfromtxt('./data/train_out.csv', delimiter=',')
test_in = np.genfromtxt('./data/test_in.csv', delimiter=',')
test_out = np.genfromtxt('./data/test_out.csv', delimiter=',')

C = []
center = np.zeros((10, 256))
n = np.zeros(10)
r = np.zeros(10)

for d in range(10):
	cloud = train_in[np.where(train_out==d)[0]]
	n[d] = cloud.shape[0]
	C.append(cloud)
	center[d] = np.mean(cloud, axis=0)
	r[d] = np.max(np.linalg.norm((cloud - center[d]), axis=1))

C = np.array(C)

dist = np.zeros((10,10))
for i in range(10):
	for j in range(10):
		dist[i,j] = np.linalg.norm(center[i] - center[j])
#print(dist)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def predict(x, m):
	N, l = x.shape
	#pred = np.zeros(N)
	d = pairwise.pairwise_distances(center, x, metric=m)
	pred=[]
	for item in d.T:
		pred.append(np.argmin(item))
	#	pred[k] = np.argmin(np.array(d))
	#	#pred[k] = np.argmin(np.linalg.norm((center - im), axis=1))
	return np.array(pred)

def accuracy(y_pred, y_true):
	mat = confusion_matrix(y_true, y_pred)
	acc = (np.sum(y_pred == y_true) /float(np.size(y_pred))) * 100
	print acc
	return acc
'''
pred_train = predict(train_in)
acc_train = accuracy(pred_train, train_out)
pred_test = predict(test_in)
acc_test = accuracy(pred_test, test_out)
'''
#print acc_train, acc_test

#plt.figure()
#plt.matshow(dist)
#plt.colorbar()
#plot_confusion_matrix(, classes=range(10), normalize=True)
#plt.show()


method = ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'] 
acc_train=np.zeros(len(method))
acc_test=np.zeros(len(method))

for i, m in enumerate(method):
	pred_train = predict(train_in,m)
	acc_train[i] = accuracy(pred_train, train_out)
	pred_test = predict(test_in, m)
	#print pred_test
	acc_test[i] = accuracy(pred_test, test_out)

ind = np.argmax(acc_test)
print acc_test[ind], method[ind]