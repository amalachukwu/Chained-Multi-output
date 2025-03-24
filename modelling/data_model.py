import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:

        y = df[Config.TYPE_COLS]
        process_y = df.y.to_numpy()
        y_series = pd.Series(process_y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        from sklearn.impute import SimpleImputer

        imp = SimpleImputer(strategy='constant', fill_value='NA')
        y_good = imp.fit_transform(y_good)

        from sklearn.preprocessing import LabelEncoder
        for i in range(4):
            y_good[:, i] = LabelEncoder().fit_transform(y_good[:, i])

        y_good = y_good.astype('int')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0)

        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X


    def get_type(self):
        return self.y
    def get_X_train(self):
        return self.X_train
    def get_X_test(self):
        return self.X_test
    def get_type_y_train(self):
        return self.y_train
    def get_type_y_test(self):
        return self.y_test
    def get_train_df(self):
        return self.train_df
    def get_embeddings(self):
        return self.embeddings
    def get_type_test_df(self):
        return self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train


class ChainedData(Data):
    """
    Extended Data class that handles chained multi-label datasets
    """
    def __init__(self, 
                 X: np.ndarray,
                 df: pd.DataFrame,
                 chain_level: str = 'type2') -> None:
        """
        Initialize chained data model with specified chain level
        
        """
        # Store the original dataframe for reference
        self.original_df = df
        
        # Get the required columns for this chain level
        self.chain_columns = Config.CHAIN_LEVELS.get(chain_level, ['y2'])
        
        # Create combined label column
        df['chained_y'] = self._create_combined_labels(df, self.chain_columns)
        
        # Set y column for the parent Data class
        df['y'] = df['chained_y']
        
        # Initialize parent class
        super(ChainedData, self).__init__(X, df)
        
        # Store the chain level for reference
        self.chain_level = chain_level
        
        # Store label mapping for evaluation
        self._create_label_mapping()
        
    def _create_combined_labels(self, df, columns):
        """
        Create combined labels from multiple columns
        """
        combined = df[columns[0]].astype(str)
        
        for col in columns[1:]:
            combined = combined + Config.LABEL_DELIMITER + df[col].astype(str)
            
        return combined
    
    def _create_label_mapping(self):
        """
        Create mapping between combined labels and individual components
        """
        self.label_mapping = {}
        
        for combined_label in self.classes:
            parts = combined_label.split(Config.LABEL_DELIMITER)
            
            # Map combined label to its components
            self.label_mapping[combined_label] = {
                col: part for col, part in zip(self.chain_columns, parts)
            }
            
    def get_chain_columns(self):
        """
        Get the columns used in this chain level
        """
        return self.chain_columns
    
    def get_label_mapping(self):
        """
        Get the mapping between combined labels and individual components
        """
        return self.label_mapping
    
    def extract_component_predictions(self, predictions):
        """
        Extract individual component predictions from combined predictions
        """
        result = {}
        
        for col in self.chain_columns:
            result[col] = []
            
        for pred in predictions:
            parts = pred.split(Config.LABEL_DELIMITER)
            
            for col, part in zip(self.chain_columns, parts):
                result[col].append(part)
                
        return result