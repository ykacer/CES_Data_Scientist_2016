import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

n_samples = 300
np.random.seed(0)
means = np.array([[0,0],[20,20]])
#C1 = np.array([[0., -0.7], [3.5, .7]])
C1 = np.array([[-2.0, 2.0], [2, -1.0]])
C2 = np.array([[1., 0.], [0., 1.]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C1) + means[0,:],
        np.dot(np.random.randn(n_samples, 2),C2) + means[1,:]]

clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X)
labels = clf.predict(X)

print "Estimated means : "
print clf.means_
print "Ground truth means : "
print means
print "Estimated covariances : "
print clf.covars_
print "Ground truth covariances : "
print C1
print C2
x = np.linspace(-20.0, 30.0,100)
y = np.linspace(-20.0, 40.0,100)
XX, YY = np.meshgrid(x, y)
# Z = np.log(-clf.eval(np.c_[XX.ravel(), YY.ravel()])[0]).reshape(XX.shape)
Z = clf.score(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
plt.close('all')

levels = np.linspace(np.amin(Z),np.amax(Z),1000)

CS = plt.contour(XX, YY, Z, levels)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
plt.axis('tight')
plt.show()
