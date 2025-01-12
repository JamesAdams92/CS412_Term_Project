# CS412_Term_Project
Classification Part

This project focused on classifying Instagram users based on their profiles and post data using machine learning. Using combined strong preprocessing, systematic hyperparameter optimization, and ensemble learning, the paper provided significant results, which discussed the importance of metadata features.

Preprocessing and Feature Engineering: Textual data from user captions underwent rigorous cleaning, including lowercasing, removal of URLs, special characters, and stopwords. Features characterizing metadata, including biography, full name, follower count, and engagement metrics (likes, comments, media type ratios), were extracted to improve the quality of the input data.

Model Building and Optimization: Text data was converted as TF-IDF with a maximum of 5000 features. GridSearchCV was used to refine the hyperparameters for three classifiers: SVM (kernel, regularization, gamma), Logistic Regression (regularization strength, solver) and Naive Bayes (smoothing parameter). This systematic tuning process ensured optimal model configurations. A Voting Classifier was used to integrate these models by making use of the complementary abilities of these models by soft voting, which yielded superior accuracy and stability than the individual classifiers.

Results and Key Insights: The Voting Classifier performed better with the highest accuracy on the validation set. Metadata examination showed user biographies, first names, and entities to be the most predictive features, as they contained important pieces of information relevant to each user's identity and interests. These results point out the utility of descriptive and engagement-related characteristics in classification studies.

Regression Part

Instagram Like Count Prediction Model Report


Objective
The purpose of this project is to predict the like count for Instagram posts using machine learning techniques. The model leverages post-specific features, user profile data, and temporal data to provide accurate estimates.

Data Preparation
Input Data
•	train-classification.csv: Contains user IDs and post categories.
–	Categories are standardized to lowercase.
–	A mapping of user IDs to categories is created.
•	training-dataset.jsonl.gz: Includes detailed user profile information, posts, captions, and follower counts.
–	Extracted into dictionaries for efficient processing:  username2posts train
and username2follower count train.
•	test-regression-round3.jsonl: Test dataset for generating predictions.

Feature Engineering
•	Temporal features derived from timestamp: hour, day of week, and month.
•	Interaction terms: hour follower interaction and day hour interaction.
•	Numeric features are normalized to the range [0, 1].
•	Squared versions of numeric features are added.
•	Missing values in caption and like count are replaced with defaults.
 
Model Architecture
Pipeline Components
•	Preprocessing:
–	Text Features: TF-IDF vectorization (max features=1000) with Turkish stop- word removal.
–	Categorical Features: OneHotEncoder for media type.
–	Numeric Features: Pass-through with interactions and squared features added.
•	Regressor: XGBoost (XGBRegressor) with reg:squarederror objective.

Custom Evaluation Metrics
•	Log Mean Squared Error (Log MSE): Log MSE = log(MSE + ϵ),	ϵ = 1 × 10−8

•	Log-MSE for Like Counts: log-MSE = 1/n ∑ (log(1+Ytrue)-log(1+Ypred))^2
 
•	Mean Difference: Mean Diff = 1/n ∑ |Ytrue - Ypred|
 
Model Optimization
•	Grid search with cross-validation (GridSearchCV) is used to tune hyperparameters:
–	n estimators: [500, 600]
–	learning rate: [0.2]
–	max depth: [9, 10]
–	subsample: [0.8]
–	colsample bytree:  [1.0]

Prediction Workflow
Input
A single Instagram post with:
•	username, caption, media type, timestamp.
 
Processing
•	Feature extraction and normalization.
•	TF-IDF and OneHot encoding.
•	Prediction using the trained pipeline and use train for predicting follower count.

Output
A rounded, non-negative estimate of the like count.

Output Generation
•	Predictions for the test dataset are generated using predict like count.
•	Results are saved in predicted like counts33.json.

Key Observations
Strengths
•	Comprehensive feature engineering, including temporal and interaction terms.
•	Custom loss functions tailored to the problem.
•	Robust hyperparameter tuning via GridSearchCV.
•	Effective handling of heterogeneous data types using pipelines. At the end I used chatGpt for preparing this readme file too.

