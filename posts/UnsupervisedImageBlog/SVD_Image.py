import numpy as np
from scipy.linalg import svd
from matplotlib import pyplot as plt

class SVD_Image():

    def svd_reconstruct(self, image, k):

        U, sigma, V = np.linalg.svd(image)

        D = np.zeros_like(image,dtype=float) # matrix of zeros of same shape as A
        D[:min(image.shape),:min(image.shape)] = np.diag(sigma)  # singular values on the main diagonal

        #A = U @ D @ V

        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]
        A_ = U_ @ D_ @ V_   

        return A_

    def svd_experiment(self, image, range):

        m, n = image.shape
        original_storage = m * n

        for k in range:
            img_rec = self.svd_reconstruct(image, k)
            storage = k * (m + n + 1)

            storage_fraction = storage / original_storage

            plt.imshow(img_rec, cmap = "Greys")
            plt.title(f'(k={k}) components, % Storage = {storage_fraction:.2f} ')
            plt.show()


