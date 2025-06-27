from processing.extract_features import extract_features
import os, pandas as pd

def run():
    rows, files, labels = [], [], []
    for root, _, fs in os.walk('dataset'):
        for f in fs:
            if f.endswith('.mp3'):
                path = os.path.join(root, f)
                label = os.path.basename(root)
                feat = extract_features(path)
                rows.append(feat)
                files.append(f)
                labels.append(label)

    df = pd.DataFrame(rows)
    df.insert(0, 'label', labels)
    df.insert(0, 'filename', files)
    df.to_csv('features/audio_features.csv', index=False)

if __name__ == '__main__':
    run()
