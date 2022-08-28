from sklearn.decomposition import PCA
import cv2
from copy import deepcopy

class PCA_3_channels():
    def __init__(self, n_components) -> None:
        self.n_components = n_components
    
    def fit(self, img):
        self.channels = cv2.split(img)
        self.pca_list = [PCA(n_components=self.n_components).fit(channel) for channel in self.channels]
        self.component_list = []
    
    def make_components(self):
        self.component_list = [pca.transform(channel) for pca, channel in zip(self.pca_list, self.channels)]
    
    def transform(self):
        self.make_components()
        return cv2.merge(self.component_list)

    def fit_transform(self, img):
        self.fit(img)
        return self.transform()

    def inverse_transform(self):
        if len(self.component_list) == 0:
            self.make_components()
        self.reconstructed_img = [pca.inverse_transform(component) for pca, component in zip(self.pca_list, self.component_list)]
        return cv2.merge(self.reconstructed_img)
    
    def inverse_transform_noise(self, noise):
        if len(self.component_list) == 0:
            self.make_components()
        noised_component = deepcopy(self.component_list)
        for i in range(len(noised_component)):
            noised_component[i] += noise[:,:,i,0]
        reconstructed_noised_img = [pca.inverse_transform(component) for pca, component in zip(self.pca_list, noised_component)]
        return cv2.merge(reconstructed_noised_img)

