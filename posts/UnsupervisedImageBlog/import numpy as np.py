import numpy as np
from scipy.linalg import svd

class LinearRegression():

    def svd_reconstruct(image, k):

        U, sigma, V = np.linalg.svd(image)

        D = np.zeros_like(A,dtype=float) # matrix of zeros of same shape as A
        D[:min(A.shape),:min(A.shape)] = np.diag(sigma)  # singular values on the main diagonal

        #A = U @ D @ V

        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]

        A_ = U_ @ D_ @ V_

        return A_