import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class ChainedModel(BaseModel):
    """
    A model wrapper that handles chained multi-output classification.
    Works with any model implementation that follows the BaseModel interface.
    """
    def __init__(self, 
                 model_class,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 chain_level: str = 'type2') -> None:
        """
        Initialize the chained model
        
        Parameters:
        -----------
        model_class: class
            The model class to use (e.g., RandomForest)
        model_name: str
            Name of the model for logging
        embeddings: np.ndarray
            Feature embeddings
        y: np.ndarray
            Target labels (should be combined for chained prediction)
        chain_level: str
            The level of chaining to use ('type2', 'type2_type3', or 'type2_type3_type4')
        """
        super(ChainedModel, self).__init__()
        self.model_class = model_class
        self.model_name = f"{model_name}_{chain_level}"
        self.embeddings = embeddings
        self.y = y
        self.chain_level = chain_level
        self.chain_columns = Config.CHAIN_LEVELS.get(chain_level, ['y2'])
        self.predictions = None
        self.component_predictions = None
        
        # Initialize the underlying model
        self.model = RandomForestClassifier()
        
    def train(self, data) -> None:
        """
        Train the chained model using the underlying model
        """
        X = data.X_train.copy()

        for i in range(3):
            y_prev = data.y_train[:, i]
            yi = data.y_train[:, i+1]

            X += np.broadcast_to(y_prev.reshape(-1, 1), (X.shape))
            self.model.fit(X, yi)

            # X_train -> 1000, 30
            # y_train -> 1000, 1 -> 1000, 30
            

        
    def predict(self, X_test, last_pred) -> np.ndarray:
        """
        Make predictions with the chained model
        """
        X = X_test.copy()
        all_preds = [last_pred]
        for i in range(3):
            X += last_pred.reshape(-1, 1)
            last_pred = self.model.predict(X)
            all_preds.append(last_pred)

        all_preds = np.stack(all_preds)
        # (3, 1000) -> (1000, 3)
        all_preds = np.transpose(all_preds)

        return all_preds

        
    def print_results(self, data):
        """
        Print results for the chained model showing both overall and component accuracy
        """
        print(f"\n--- Chained Model Results for {self.chain_level} ---")
        
        y_true = np.array(data.y_test)
        y_pred = self.predict(data.X_test, data.y_test[:, 0])

        
        # Check input shapes
        if y_true.shape != y_pred.shape or y_true.shape[1] != 4:
            raise ValueError(f"Expected arrays of shape (n, 4), got {y_true.shape} and {y_pred.shape}")
        
        # Calculate accuracy for each sample
        n_samples = y_true.shape[0]
        accuracies = np.zeros(n_samples)
        
        for i in range(n_samples):
            correct_count = 0
            all_correct = True
            
            # Check each prediction sequentially
            for j in range(4):
                if all_correct and y_true[i, j] == y_pred[i, j]:
                    correct_count += 1
                else:
                    all_correct = False  # Once a prediction is wrong, subsequent ones don't matter
            
            # Calculate accuracy as percentage of correct predictions
            accuracies[i] = (correct_count / 3) * 100
        
        # Return mean accuracy across all samples
        self.accuracy = np.mean(accuracies)
        print(self.accuracy)
        return self.accuracy

            
    def data_transform(self) -> None:
        """
        Implement data transform for BaseModel compatibility
        """
        if hasattr(self.model, 'data_transform'):
            self.model.data_transform()
        else:
            pass  # Default implementation
            
    def get_component_predictions(self):
        """
        Get the predictions for individual components
        """
        return self.component_predictions



