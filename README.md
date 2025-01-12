# CS412_Term_Project
Classification Part

This project focused on classifying Instagram users based on their profiles and post data using machine learning. Using combined strong preprocessing, systematic hyperparameter optimization, and ensemble learning, the paper provided significant results, which discussed the importance of metadata features.

Preprocessing and Feature Engineering: Textual data from user captions underwent rigorous cleaning, including lowercasing, removal of URLs, special characters, and stopwords. Features characterizing metadata, including biography, full name, follower count, and engagement metrics (likes, comments, media type ratios), were extracted to improve the quality of the input data.

Model Building and Optimization: Text data was converted as TF-IDF with a maximum of 5000 features. GridSearchCV was used to refine the hyperparameters for three classifiers: SVM (kernel, regularization, gamma), Logistic Regression (regularization strength, solver) and Naive Bayes (smoothing parameter). This systematic tuning process ensured optimal model configurations. A Voting Classifier was used to integrate these models by making use of the complementary abilities of these models by soft voting, which yielded superior accuracy and stability than the individual classifiers.

Results and Key Insights: The Voting Classifier performed better with the highest accuracy on the validation set. Metadata examination showed user biographies, first names, and entities to be the most predictive features, as they contained important pieces of information relevant to each user's identity and interests. These results point out the utility of descriptive and engagement-related characteristics in classification studies.
