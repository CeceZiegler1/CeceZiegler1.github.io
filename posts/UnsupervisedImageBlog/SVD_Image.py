import numpy as np
from scipy.linalg import svd
from matplotlib import pyplot as plt

class SVD_Image():

    def svd_reconstruct(self, image, k):

        #input: a matrix as an image, k value for number of components to use to reconstruct image
       
        #This method uses singular value decomposition to reconstruct an image
        
        #output: a matrix, A_ that is the reconstructed image with k components

        #https://middlebury-csci-0451.github.io/CSCI-0451/assignments/blog-posts/blog-post-image-processing.html
        #I give credit for code I used from the link above, the assignment outline, in my function

        U, sigma, V = np.linalg.svd(image)

        D = np.zeros_like(image,dtype=float) # matrix of zeros of same shape as A
        D[:min(image.shape),:min(image.shape)] = np.diag(sigma)  # singular values on the main diagonal


        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]
        A_ = U_ @ D_ @ V_   #reconstructed image

        return A_

    def svd_experiment(self, image, range):

        #input: matrix as image, range of values for k that gives start, end and step size
        
        #this method uses a for loop to display different reconstructed images with different k values
        #along with computed the %storage that each image takes up
        
        #output: no return value

        m, n = image.shape #getting the dimensions of image to calculate storage
        original_storage = m * n #storage of non reconstructed image

        for k in range:
            img_rec = self.svd_reconstruct(image, k)
            storage = k * (m + n)

            storage_fraction = (storage / original_storage) *100 #computing percent of storage

            plt.imshow(img_rec, cmap = "Greys")#displaying image
            plt.title(f'(k={k}) components, % Storage = {storage_fraction:.2f} ')
            plt.show()


