import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from processing.extract_features import extract_features

def find_top_3(file_path, csv_path='features/audio_features.csv'):
    input_feat = extract_features(file_path).reshape(1, -1)
    df = pd.read_csv(csv_path)
    feats = df.iloc[:, 2:].values
    sims = cosine_similarity(input_feat, feats)[0]
    top_idxs = sims.argsort()[-3:][::-1]
    return [(df.iloc[i]['filename'], df.iloc[i]['label'], sims[i]) for i in top_idxs]
