# AEDLSGAN

Official implementation of AEDLSGAN  

Environment: Windows 10, Tensorflow 2.7  

To train the model:
1. Set Hyperparameters.py, then run Main.py

To generate samples from pre-trained model:
1. Download discriminator.h5 and generator.h5 from the releases, then put .h5 files to 'models'.  
2. Set train_data_size=0, load_model=True, evaluate_model=False in Hyperparameters.py, then run Main.py.  

Algorithm for AEDLSGAN is in "Train.py"  
