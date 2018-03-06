import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import math

n_samples = 300
dimension_de_x = 2
k = 2

def compute_ll(data, mu, sigma, phi, K):
    W = np.random.randn(len(data), 2)
    W = Expectation(data, sigma,mu, phi)
    ll = np.zeros((len(data), 1))
    for i in range(len(data)):
        sumi=0
        for j in range(K):
            determinant = np.absolute(np.linalg.det(sigma[j]))
            dataligne = np.matrix(np.array([data[i,]])) - np.matrix(np.array([mu[j,]]))
            transpose = dataligne.transpose()
            num = 1 / ((2 * math.pi) * math.sqrt(determinant))
            num = num * math.exp((-1 / 2) * ((dataligne) * np.linalg.inv(sigma[j]) * transpose)) * phi[0, j]
            if W[i, j] != 0:
                sumi = sumi + W[i, j] * np.log(num / W[i, j])
        ll[i] = sumi
    #print(ll.shape)
    return ll


def Expectation(data, sigma, mu, phi):
    W = np.zeros((len(data), 2))
    for i in range(len(data)):
        for j in range(2):
            # numerateur
            determinant = np.absolute(np.linalg.det(sigma[j]))
            dataligne = np.matrix(np.array([data[i,]])) - np.matrix(np.array([mu[j,]]))
            transpose = dataligne.transpose()

            num = 1 / ((2 * math.pi) * math.sqrt(determinant))
            num = num * math.exp((-1 / 2) * ((dataligne) * np.linalg.inv(sigma[j]) * transpose)) * phi[0, j]

            # denominateur
            determinant = np.absolute(np.linalg.det(sigma[0]))
            dataligne = np.matrix(np.array([data[i,]])) - np.matrix(np.array([mu[0,]]))
            transpose = dataligne.transpose()

            den1 = (1 / ((2 * math.pi) * math.sqrt(determinant)))
            den1 = den1 * math.exp((-1 / 2) * ((dataligne) * np.linalg.inv(sigma[0]) * transpose)) * phi[0, 0]

            determinant = np.absolute(np.linalg.det(sigma[1]))
            dataligne = np.matrix(np.array([data[i,]])) - np.matrix(np.array([mu[1,]]))
            transpose = dataligne.transpose()

            den2 = (1 / ((2 * math.pi) * math.sqrt(determinant)))
            den2 = den2 * math.exp((-1 / 2) * ((dataligne) * np.linalg.inv(sigma[1]) * transpose)) * phi[0, 1]

            den = den1 + den2
            with np.errstate(invalid='ignore'):
                W[i, j] = num / den

    return W


def Maximisation_phi(w, phi):
    sommekluster1 = 0
    sommekluster2 = 0

    for i in range(600):
        # print("la valeur en i ",i,"et j 1 est ",w[i,0])
        sommekluster1 = sommekluster1 + w[i, 0]

        # print("la valeur en i ",i,"et j 2 est ",w[i,1])
        sommekluster2 = sommekluster2 + w[i, 1]

    phi[0, 0] = sommekluster1 / 600
    phi[0, 1] = sommekluster2 / 600

    return phi


def Maximisation_mu(w, data, mu):
    sommekluster1 = 0
    sommekluster2 = 0

    sommeproba1 = 0
    sommeproba2 = 0

    for i in range(600):
        # print(dataligne)
        sommekluster1 = sommekluster1 + w[i, 0] * data[i]
        sommekluster2 = sommekluster2 + w[i, 1] * data[i]

        sommeproba1 = sommeproba1 + w[i, 0]
        sommeproba2 = sommeproba2 + w[i, 1]

    mu[0] = sommekluster1 / sommeproba1
    mu[1] = sommekluster2 / sommeproba2

    return mu


def Maximisation_sigma(w, data, mu, sigma):
    sommekluster1 = 0
    sommekluster2 = 0
    temp1 = 0
    temp2 = 0

    sommeproba1 = 0
    sommeproba2 = 0

    for i in range(600):
        dataligne = [data[i,]]
        dataligne = np.matrix(np.array(dataligne))
        dataligne = dataligne - np.matrix(np.array([mu[0,]]))
        transpose = dataligne.transpose()

        temp1 = w[i, 0] * transpose * dataligne

        sommekluster1 = sommekluster1 + temp1
        sommeproba1 = sommeproba1 + w[i, 0]

        sigma[0] = sommekluster1 / sommeproba1

        dataligne = [data[i,]]
        dataligne = np.matrix(np.array(dataligne))
        dataligne = dataligne - np.matrix(np.array([mu[1,]]))
        transpose = dataligne.transpose()

        temp2 = w[i, 1] * transpose * dataligne

        sommekluster2 = sommekluster2 + temp2
        sommeproba2 = sommeproba2 + w[i, 1]

        sigma[1] = sommekluster2 / sommeproba2

    return sigma


# generate random sample, two components
np.random.seed(1)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, dimension_de_x) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, dimension_de_x), C)

# probablities
w = np.random.randn(600, 2)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
n = 2
J = 2
K = 2
old_ll = 0
new_ll = 0
tol = 0.001

sigma = np.array([np.random.rand(k, dimension_de_x), np.random.rand(k, dimension_de_x)])
mu = np.array([np.random.rand(1, dimension_de_x), np.random.rand(1, dimension_de_x)])
phi = np.matrix([[0.2, 0.8]])

for iteration in range(10):
    old_ll = np.sum(compute_ll(X_train, mu, sigma, phi,K))
    test = Expectation(X_train, sigma, mu, phi)
    w= test
    phi = Maximisation_phi(w, phi)
    mu = Maximisation_mu(w, X_train, mu)
    sigma = Maximisation_sigma(w, X_train, mu, sigma)
    # Likelihood
    new_ll = np.sum(compute_ll(X_train, mu, sigma, phi,K))

    if (np.abs(new_ll - old_ll) < tol):
        print("Convergence got in the iteration number:", iteration + 1)
        break
print(" ")
x = np.linspace(-20., 30.)
y = np.linspace(-20., 40.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -compute_ll(XX, mu, sigma, phi,K)
Z = Z.reshape(X.shape)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
print("Valeur de mu:")
print(mu)
print("")
print("Valeurs de w:")
print(w)
print("")
print("Valeur de sigma:")
print(sigma)
plt.show()
