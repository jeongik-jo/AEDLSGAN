# AEDLSGAN

Official implementation of AEDLSGAN  

Environment: Windows 11, Tensorflow 2.10  

To train the model:
1. Set Hyperparameters.py, then run Main.py

To generate samples from pre-trained model:
1. Download discriminator.h5 and generator.h5 from the releases, then put .h5 files to 'models'.  
2. Set learning_rate=0.0, decay=0.0, latent_var_decay_rate=1.0, train_data_size=batch_size, shuffle_test_dataset=True, load_model=True, evaluate_model=False in Hyperparameters.py, then run Main.py.  

Algorithm for AEDLSGAN is in "Train.py"  
