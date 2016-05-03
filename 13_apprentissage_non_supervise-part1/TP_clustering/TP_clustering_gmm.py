import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples1 = 1000
n_samples2 = 800
np.random.seed(0)
means = np.array([[0,0],[20,20]])
C1 = np.array([[1., 0.], [0., 1.]])
C2 = np.array([[1.0, 0.], [0., 10.0]])
X = np.r_[np.dot(np.random.randn(n_samples1, 2), C1) + means[0,:],
        np.dot(np.random.randn(n_samples2, 2),C2) + means[1,:]]
X = np.r_[np.random.multivariate_normal(means[0,:], C1, n_samples1),np.random.multivariate_normal(means[1,:], C2, n_samples2)]
print X.shape

clf = mixture.GMM(n_components=2, covariance_type='full',n_iter=1000)
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
print "estimated weights : "
print clf.weights_
print "grond truth weights : "
print np.array([1.*n_samples1/(n_samples1+n_samples2),1.*n_samples2/(n_samples1+n_samples2)])
x = np.linspace(-20.0, 30.0,100)
y = np.linspace(-20.0, 40.0,100)
XX, YY = np.meshgrid(x, y)
# Z = np.log(-clf.eval(np.c_[XX.ravel(), YY.ravel()])[0]).reshape(XX.shape)
# Z = clf.score(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
Z = -clf.score_samples(np.c_[XX.ravel(), YY.ravel()])[0]
Z = Z.reshape(XX.shape)
plt.close('all')

CS = plt.contour(XX, YY, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),levels=np.logspace(0,3,10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.title('Negative log-likelihood predicted by a GMM')
plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
plt.axis('tight')
plt.show()
