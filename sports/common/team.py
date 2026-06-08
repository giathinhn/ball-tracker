import numpy as np
from sklearn.cluster import KMeans
from typing import List

class TeamClassifier:
    """
    A classifier that uses the average color of the upper half of the player crop
    for jersey representation and KMeans for clustering into two teams.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract the average RGB/BGR color of the upper portion of the crops (jersey region).
        """
        features = []
        for crop in crops:
            if crop is None or crop.size == 0:
                features.append([0.0, 0.0, 0.0])
                continue
            h, w = crop.shape[:2]
            
            # Crop to the upper body (10% to 55% of height) to focus on the shirt
            y_start = int(h * 0.1)
            y_end = int(h * 0.55)
            shirt_region = crop[y_start:y_end, :]
            
            if shirt_region.size == 0:
                features.append([0.0, 0.0, 0.0])
                continue
                
            avg_color = np.mean(shirt_region, axis=(0, 1))
            features.append(avg_color)
            
        return np.array(features)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the KMeans model using the extracted average color vectors.
        """
        data = self.extract_features(crops)
        if len(data) > 0:
            self.cluster_model.fit(data)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the team ID based on the closest color cluster.
        """
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        return self.cluster_model.predict(data)
