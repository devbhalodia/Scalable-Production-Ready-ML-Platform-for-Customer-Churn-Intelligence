import os
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from preprocessing import TextStandardizer

def train(input_path, output_path, args):
    df = pd.read_parquet(os.path.join(input_path, "clean_data.parquet"))

    df['attrition_flag'] = df['attrition_flag'].map({'existing customer': 0, 'attrited customer': 1})
    X = df.drop(columns=['attrition_flag'])
    y = df['attrition_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    education_order = ['unknown', 'uneducated','high school','college' ,'graduate','post-graduate', 'doctorate']
    income_order = ["unknown","less than $40k","$40k - $60k","$60k - $80k","$80k - $120k", "$120k +"]
    card_order = ["blue", "silver","gold","platinum"]

    cat_pipeline_1 = Pipeline([
        ("text_standardize", TextStandardizer()),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    cat_pipeline_2 = Pipeline([
        ("text_standardize", TextStandardizer()),
        ("ordinal_encode", OrdinalEncoder(categories=[education_order, income_order, card_order], 
                                          handle_unknown="use_encoded_value", unknown_value=-1))])

    col_transformer = ColumnTransformer(
        transformers=[
            ('num_pip', 'passthrough', ['customer_age', 'dependent_count', 'months_on_book', 'total_relationship_count', 'months_inactive_12_mon', 'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal', 'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt', 'total_trans_ct', 'total_ct_chng_q4_q1', 'avg_utilization_ratio']),
            ('cat_pip_1', cat_pipeline_1, ['gender', 'marital_status']),
            ('cat_pip_2', cat_pipeline_2, ['education_level', 'income_category', 'card_category'])
        ]
    )

    xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    subsample=args.subsample,
    colsample_bytree=args.colsample_bytree,
    scale_pos_weight=(len(y_train[y_train == 0]) / len(y_train[y_train == 1])),
    tree_method="hist",
    eval_metric="logloss"
    )

    full_pipeline = Pipeline([("preprocess", col_transformer), ("model", xgb_model)])
    
    full_pipeline.fit(X_train, y_train)

    model_file_path = os.path.join(output_path, "model.joblib")
    joblib.dump(full_pipeline, model_file_path)
    print(f"Model saved to {model_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)

    args = parser.parse_args()
    
    train(args.train, args.model_dir, args)