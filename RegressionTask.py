
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import json
import gzip
from pprint import pprint
#@title Turkish StopWords
import nltk
from nltk.corpus import stopwords
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split



nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

train_classification_df = pd.read_csv("train-classification.csv",)
train_classification_df = train_classification_df.rename(columns={'Unnamed: 0': 'user_id', 'label': 'category'})

# Unifying labels
train_classification_df["category"] = train_classification_df["category"].apply(str.lower)
username2_category = train_classification_df.set_index("user_id").to_dict()["category"]

# stats about the labels
train_classification_df.groupby("category").count()

train_data_path = "training-dataset.jsonl.gz"

username2posts_train = dict()
username2follower_count_train = dict()
username2caption_train = dict()





with gzip.open(train_data_path, "rt") as fh:
  for line in fh:
    sample = json.loads(line)

    profile = sample["profile"]
    username = profile["username"]
    follower_count = profile.get("follower_count" , "")
    



    username2follower_count_train[username] = follower_count
    
   
    username2posts_train[username] = sample["posts"]
     





print("finished p1")

data = []


for username, posts in username2posts_train.items():
    

    for post in posts:
        media_type = post.get("media_type", "UNKNOWN")
        timestamp = post.get("timestamp")
        like_count = post.get("like_count", 0)
        caption = post.get("caption", "")

       
        data.append({
            "follower_count": follower_count,
            "media_type": media_type,
            "timestamp": timestamp,
            "caption": caption if caption is not None else "", 
            "like_count": like_count,
        })
print("finished p2")

df = pd.DataFrame(data)



df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['hour_follower_interaction'] = df['hour'] * df['follower_count']
df['day_hour_interaction'] = df['day_of_week'] * df['hour']


df = df.drop(columns=['timestamp'])



epsilon = 1e-8

numeric_features = ['follower_count', 'hour', 'day_of_week', 'month', 'hour_follower_interaction', 'day_hour_interaction'] 
df[numeric_features] = (df[numeric_features] - df[numeric_features].min()) / \
                       (df[numeric_features].max() - df[numeric_features].min() + epsilon)


for feature in numeric_features:
    df[f"{feature}_squared"] = df[feature] ** 2

df['caption'] = df['caption'].fillna("")
df['like_count'] = df['like_count'].fillna(0)





X = df.drop(columns=['like_count'])
y = df['like_count']


tfidf = TfidfVectorizer(max_features=1000, stop_words=turkish_stopwords, ngram_range=(1, 2))
onehot = OneHotEncoder(handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf, 'caption'),
        ('onehot', onehot, ['media_type']),
    ],
    remainder='passthrough'  
   , force_int_remainder_cols=False 
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def log_mse(y_true, y_pred):

   
    raw_y_true = np.expm1(y_true)  
    raw_y_pred = np.expm1(y_pred)  

  
    raw_y_true = np.round(raw_y_true)
    raw_y_pred = np.round(raw_y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    return np.log(mse + 1e-8)

def log_mse_like_counts(y_true, y_pred):
    
    raw_y_true = np.expm1(y_true)  
    raw_y_pred = np.expm1(y_pred)  

    
    raw_y_true = np.round(raw_y_true)
    raw_y_pred = np.round(raw_y_pred)
    
    log_y_true = np.log1p(y_true)
    log_y_pred = np.log1p(y_pred)
    squared_errors = (log_y_true - log_y_pred) ** 2
    return np.mean(squared_errors)

def mean_diff(y_true, y_pred):
    
    
    raw_y_true = np.expm1(y_true) 
    raw_y_pred = np.expm1(y_pred)  

   
    raw_y_true = np.round(raw_y_true)
    raw_y_pred = np.round(raw_y_pred)
    print(abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred))


log_mse_scorer = make_scorer(log_mse, greater_is_better=False)
log_mse_like_counts_scorer = make_scorer(log_mse_like_counts, greater_is_better=False)
mean_diff_scorer = make_scorer(mean_diff, greater_is_better=False)



y_train_transformed = np.log1p(y_train) 
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
])


param_grid = {
    'regressor__n_estimators': [ 500 , 600],
    'regressor__learning_rate': [  0.2],
    'regressor__max_depth': [  9 , 10 ],
    'regressor__subsample': [ 0.8],
    'regressor__colsample_bytree': [ 1.0],
}


grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring=log_mse_like_counts_scorer, verbose=3)


grid_search.fit(X_train, y_train_transformed)




model_pipeline = grid_search.best_estimator_


y_test_transformed = np.log1p(y_test)  
y_pred_transformed = model_pipeline.predict(X_test)


log_mse_score = log_mse(y_test_transformed, y_pred_transformed)
log_mse_like_counts_score = log_mse_like_counts(y_test_transformed, y_pred_transformed)
mean_diff_score_value = mean_diff(y_test_transformed, y_pred_transformed)



print(f"Log MSE: {log_mse_score}")
print(f"Log MSE Like Counts: {log_mse_like_counts_score}")
print(f"Mean Difference: {mean_diff_score_value}")

def predict_like_count(username, post, timestamp):
    caption = post.get("caption", "")  
    if caption is None:  
        caption = ""
    media_type = post.get("media_type", "UNKNOWN")
    follower_count = username2follower_count_train.get(username, 0)


    test_data = {
        "follower_count": follower_count,
        "media_type": media_type,
        "caption": caption,  
        "timestamp": timestamp,  
    }


    test_df = pd.DataFrame([test_data])

    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df['hour'] = test_df['timestamp'].dt.hour
    test_df['day_of_week'] = test_df['timestamp'].dt.dayofweek
    test_df['month'] = test_df['timestamp'].dt.month
    test_df['hour_follower_interaction'] = test_df['hour'] * test_df['follower_count']
    test_df['day_hour_interaction'] = test_df['day_of_week'] * test_df['hour']

    
    test_df = test_df.drop(columns=['timestamp'])

    
    epsilon = 1e-8
    numeric_features = ['follower_count', 'hour', 'day_of_week', 'month', 'hour_follower_interaction', 'day_hour_interaction']
    test_df[numeric_features] = (test_df[numeric_features] - test_df[numeric_features].min()) / \
                                 (test_df[numeric_features].max() - test_df[numeric_features].min() + epsilon)

    for feature in numeric_features:
        test_df[f"{feature}_squared"] = test_df[feature] ** 2

    test_df['caption'] = test_df['caption'].fillna("")
    
    processed_features = model_pipeline.named_steps['preprocessor'].transform(test_df)

   
    predicted_log_like_count = model_pipeline.named_steps['regressor'].predict(processed_features)[0]
    
   
    predicted_like_count = np.expm1(predicted_log_like_count)
    return max(0, int(predicted_like_count))  

    
output_dict = {}

path = "test-regression-round3.jsonl"
output_path = "predicted_like_counts33.json"


with open(path, "rt") as fh:
    for line in fh:
        sample = json.loads(line)
        username = sample["username"]
        user_id = sample["id"]  
        predicted_like_count = predict_like_count(username, sample, sample["timestamp"])
        output_dict[user_id] = predicted_like_count


with open(output_path, "w") as of:
    json.dump(output_dict, of, indent=4)


pprint(list(output_dict.items())[:3])


