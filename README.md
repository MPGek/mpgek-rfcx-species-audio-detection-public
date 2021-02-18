# mpgek-rfcx-species-audio-detection-public
22nd place solution for Rainforest Connection Species Audio Detection - https://www.kaggle.com/c/rfcx-species-audio-detection/

To train and predict models you can run script of TP and FP: run_scripts/train_model_simple.py
To train FP co-teaching run run_scripts/train_model_coteaching.py

Dataset content from the Kaggle should be placed in the folder 'data'.

All training configuration are placed in the folder 'configs'.

In training script in the main function you can find variable 'cfg'. To train different experiments change this variable with content of files inside the folder 'configs'.
