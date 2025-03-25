from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from model.randomforest import RandomForest
import random
seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load input data
    df = get_input_data()
    return df

def preprocess_data(df):
    # Corresponding input data.
    df = de_duplication(df)
    # Remove noisy input data.
    df = noise_remover(df)
    # Translate data to english.
   
    return df

def get_embeddings(df:pd.DataFrame):
    # Gets tf-idf embeddings.
    X = get_tfidf_embd(df)  
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    """
    Perform traditional single-label classification
    """
    model_predict(data, df, name)
    
def perform_chained_modelling(data: Data, df: pd.DataFrame):
    """
    Perform chained multi-label classification
    """
    chain_model_predict(data, df, RandomForest, "RandomForest")

if __name__ == '__main__':
    # Pre-processing steps:
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Data transformation.
    X, group_df = get_embeddings(df)
    
    # Create data object.
    data = get_data_object(X, df)

    print(data.y_train.shape)
    
    
    
    # Multi-label classification with chained approach.
    perform_chained_modelling(data, df)