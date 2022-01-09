from matplotlib import axes
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)  
    return x-np.mean(x, axis=0) 

def get_covariance(dataset):
    r = np.dot(dataset.T, dataset)
    r = r / (len(dataset) - 1)
    return r

def get_eig(S, m):
    value, vector = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    val = value.argsort()[::-1]
    vec = vector[:, val]
    return np.diag(value[val]), vec

def get_eig_perc(S, perc):
    eigen_val = eigh(S, eigvals_only=True)[::-1]
    total_val = np.sum(eigen_val)
    m = 0
    for i in range(len(eigen_val)):
        if eigen_val[i]/total_val > perc:
            m+=1
    return get_eig(S,m)

def project_image(img, U):
    a = np.zeros(shape=(len(U[0]),1))
    for i in range(len(U[0])):
        a[i] = np.dot(U.T[i], img)
    projection = 0
    for i in range(len(a)):
        projection += a[i]*U[:,i]
    return projection

def display_image(orig, proj):
    orig = np.reshape(orig, newshape=(32,32))
    proj = np.reshape(proj, newshape=(32,32))
    fig, ax = plt.subplots(1,2)
    o = ax[0].imshow(orig.T)
    p = ax[1].imshow(proj.T)
    ax[0].set_title('Original')
    ax[1].set_title('Projection')
    fig.colorbar(o, ax=ax[0])
    fig.colorbar(p, ax=ax[1])
    plt.show()
    return 0