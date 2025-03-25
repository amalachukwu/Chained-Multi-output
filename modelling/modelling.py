from model.randomforest import RandomForest
from model.chained_model import ChainedModel
from modelling.data_model import ChainedData
from Config import *

def model_predict(data, df, name):
    """
    Regular model prediction for single-label classification
    """
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def chain_model_predict(data, df, model_class, model_name):
    """
    Chained model prediction for multi-label classification
    """
    results = []
    X = data.get_embeddings()
    
    print(f"\n=== Chained Multi-Output Classification with {model_name} ===")
    
    # Create models for each chaining level.
    for chain_level in Config.CHAIN_LEVELS.keys():
        print(f"\nProcessing chain level: {chain_level}")
        
        # Create chained data for this level.
        chained_data = ChainedData(X, df, chain_level)
        
        if chained_data.X_train is None:
            print(f"Skipping {chain_level} due to insufficient data...")
            continue
            
        # Create and train chained model.
        model = ChainedModel(model_class, model_name, chained_data.get_embeddings(), chained_data.get_type(), chain_level)
        model.train(chained_data)
        model.predict(chained_data.X_test, chained_data.y_test[:, 0])
        model.print_results(chained_data)
        
        # Store results.
        results.append({
            'chain_level': chain_level,
            'model': model,
            'data': chained_data
        })
    
    return results


def model_evaluate(model, data):
    """
    Evaluate a model against data
    """
    model.print_results(data)