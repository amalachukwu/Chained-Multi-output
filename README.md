# **Multi-Label Email Classification System**
This project implements a modular architecture for multi-label email classification with two architectural approaches: Chained Multi-outputs and Hierarchical Modeling.

#  Project Overview*
This system is designed to classify emails with multiple dependent variables (Type 2, Type 3, and Type 4) while adhering to software engineering principles of modularity, consistency, and abstraction.

# Key Features*
Modular Architecture: Separation between preprocessing and modeling components
Consistent Data Handling: Standard interfaces for data across models
Abstraction: Common interface for all model implementations
Multi-label Support: Two distinct approaches for multi-label classification

# Architectural Approaches*
1. Chained Multi-Output Approach
In this approach, one model instance assesses multiple types in a chained manner:
Type 2
Type 2 + Type 3
Type 2 + Type 3 + Type 4

The model evaluates accuracy at each level of the chain.
2. Hierarchical Modeling Approach
In this approach, multiple model instances are created in a hierarchical manner:

A base model classifies Type 2
For each Type 2 class, a specialized model classifies Type 3
For each Type 2+Type 3 combination, a specialized model classifies Type 4

This allows for more specialized predictions based on previous classifications.

# Requirements*
Python 3.8+
NumPy
Pandas
scikit-learn

Running the Project

Clone the repository
Install dependencies: pip install -r requirements.txt
Place your data files in the data/ directory
Run the main script:
python main.py


# Configuration
You can modify Config.py to change various settings:

Input data columns
Types to classify
Chain levels for multi-label classification

# Implementation Details
Chained Multi-Output Implementation
The chained approach combines labels at different levels and trains models to predict these combined labels. The ChainedData class handles creating and managing combined labels, while the ChainedModel class manages the training and evaluation process.
Evaluation
Accuracy is calculated hierarchically, meaning that if a prediction for an earlier type is incorrect, predictions for subsequent types are considered incorrect as well. This reflects the real-world dependency between different label types.

# Authors
Dhruva Deswal 
Amalachukwu Adaeze Atusiuba

