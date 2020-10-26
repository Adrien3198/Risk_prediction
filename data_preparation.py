import pandas as pd
from sklearn.preprocessing import LabelEncoder

"""
data_preparation.py
===================
"""
def preprocess(dataset="datasets/application_train.csv"):
    """
        Return the dataset preprocessed (pandas Dataframe)

        Parameters
        ----------
        dataset
            Path to the dataset content .csv

    """
    df = pd.read_csv(dataset)
    del_cols = [col for col in df if (df[col].isna().sum()/df[col].size) > 0.2]
    df = df.drop(columns=del_cols)

    cat_vars = [c for c in df if df[c].dtype == "object"]
    df = df.dropna(axis=0, how='any', subset=cat_vars)

    le = LabelEncoder()

    label_encoded_vars = []
    dummies_vars = []
    for cat in cat_vars:
        if df[cat].nunique() <= 2:
            df[cat] = le.fit_transform(df[cat])
            label_encoded_vars.append(cat)
            
        else:
            dummies_vars.append(cat)
    df_cat = pd.concat([
            df[label_encoded_vars], 
            pd.get_dummies(df[dummies_vars])
        ],axis=1)

    df_num = df.drop(columns=cat_vars)

    df_num = df_num.fillna(df_num.mean())
    df = pd.concat([df_num, df_cat], axis=1)
    return df

if __name__ == "__main__":
    preprocess()
