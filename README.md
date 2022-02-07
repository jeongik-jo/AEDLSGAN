# AEDLSGAN

Official implementation of AEDLSGAN  

Environment: Windows 10, Tensorflow 2.7  

To train model: edit HyperParameters.py, and run Main.py  

There are trained model with current hyperparameters setting in 'models' folder. To generate samples from trained model: set train_data_size=0, load_model=True, evaluate_model=False in HyperParameters.py, and run Main.py  

Algorithm for AEDLSGAN is in "Train.py"  
