# vesselsegmentation
This is a deep learning-based pipline for automatically extracting vessels in 3D microscopy images from human glioma cleared samples. The publication about this source code is 'Three-dimensional visualization of blood vessels in human gliomas based on tissue clearing and deep learning'. The code contains four parts: preprocessing, 3D U-Net segmentation, post-processing, and quantification of 3D vascular images.

### part1 preprocessing
Store the raw data in ./data/data_raw and run ./code/preprocessing.py. The results will be saved in ./data/data_pre.

### part2 3D U-Net segmentation
You can create your own model folder in ./3dunet_results. 

During the training phase, store the training data and labels in ./data/data_pre and ./data/data_label. Run ./code/train.py after setting suitable parameters. The network model files and training indicators will be saved in ./3dunet_results/YourModelName/model/YourTrainingDataName and ./3dunet_results/YourModelName/record/YourTrainingDataName. 

During the testing phase, store the testing data and labels (not essential) in ./data/data_pre and ./data/data_label. Choose a trained model file and Run ./code/test.py. The predicted results will be saved in ./3dunet_results/YourModelName/prediction/YourTestingDataName.

### part3 post-processing
Run ./code/postprocess.py to post-process the predicted volumes. The results of denoising and closing will saved in ./3dunet_results/YourModelName/prediction/YourTestingDataName.

### part4 quantification
Run ./code/feats.py to extract centerline, radius, and branching point features and the metrics including mean radius and number of branching points will be saved in
./3dunet_results/YourModelName/prediction/YourTestingDataName/quantify. Other metrics can be measured by running ./code/measure_vessels.m in Matlab.

