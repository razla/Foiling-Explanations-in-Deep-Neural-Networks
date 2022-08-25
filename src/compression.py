from sklearn.decomposition import PCA
import cv2

class PCA_3_channels():
    def __init__(self, n_components) -> None:
        self.n_components = n_components
    
    def fit(self, img):
        self.channels = cv2.split(img)
        self.pca_list = [PCA(n_components=self.n_components).fit(channel) for channel in self.channels]
    
    def transform(self):
        self.component_list = [pca.transform(channel) for pca, channel in zip(self.pca_list, self.channels)]
        return cv2.merge(self.component_list)

    def fit_transform(self, img):
        self.fit(img)
        return self.transform()

    def inverse_transform(self):
        self.reconstructed_img = [pca.inverse_transform(component) for pca, component in zip(self.pca_list, self.component_list)]
        return cv2.merge(self.reconstructed_img)
