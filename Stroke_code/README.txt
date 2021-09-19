This repository contains:
Data/Soroka folder with 3 sub-folders:
code_products: products of this code
organized_train_data: data set with same structure as the original data but as csv files for Immediate use (without special reading)
raw_data: all original data, including matlab features from the paper experiment and movements classification.

code folder:
feature_Selection.py: this is the "main" file for you to play with. currently there is no feature selection - but you should add feature selection there.

baseline_Rakel_in_python.py: same as the "R" code, the experiment from the paper.
create_tsfresh_features.py: tsfresh feature creation from organized and preprocessed data.
label_creation.py: create label file for both train and test data. *you should Run it once.
pre_processing.py: all preprocessing utils for the data.
read_stroke.py: read raw data and creates organized data *you should Run it once for train and once for test it you use the test.
utils.py: some utils that used in the code.

TO RUN THE CODE (First time)- Leave One Out on the original data:
run read_stroke.py
run label_creation.py
run feature_Selection.py

after that you can comment row 164 in feature_Selection.py and run feature_Selection.py

If you want to use the new test set:
* comment rows 107 - 112 in read_stroke.py
* uncomment rows 116-118 in read_stroke.py
run read_stroke.py
* comment rows 178 in feature_Selection.py
* uncomment rows 171-176 in feature_Selection.py
run feature_Selection.py



