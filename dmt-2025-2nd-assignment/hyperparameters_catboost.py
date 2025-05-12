pip install catboost

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# loading model
model = CatBoostClassifier()
model.load_model('dmt-2025-2nd-assignment/catboost_model.cbm')  

# look at hyperparameters
params = model.get_params()
print("Hyperparameters:\n", params)

# Feature importance plots
try:
    importance = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance)
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.show()
except:
    print("Error")
