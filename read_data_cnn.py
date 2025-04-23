import os
import librosa
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# Caminhos
zip_dir = r"C:\Users\netog\Downloads\3670185"
meta_file = r"C:\Users\netog\Downloads\3670185\meta.csv"

# Leitura do arquivo de metadados
meta = pd.read_csv(meta_file, sep='\t')
meta_dict = dict(zip(meta['filename'], meta['scene_label']))
zip_files = sorted([f for f in os.listdir(zip_dir) if f.endswith('.zip') and 'audio' in f])

# Parâmetros MFCC
sr = 22050
n_mfcc = 13

# Listas para DataFrame
filenames, labels, features = [], [], []

# Processa os arquivos
for zip_name in tqdm(zip_files):
    with zipfile.ZipFile(os.path.join(zip_dir, zip_name), 'r') as z:
        for name in z.namelist():
            filename = name.split('/')[-1]
            if not filename.endswith('.wav'):
                continue
            try:
                with z.open(name) as wav_file:
                    y, sr_native = sf.read(wav_file)
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)  # converte para mono

                    y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)

                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    delta = librosa.feature.delta(mfcc)
                    delta2 = librosa.feature.delta(mfcc, order=2)

                    # Stack para CNN: shape = (3, 13, T)
                    feature_tensor = np.stack([mfcc, delta, delta2])  # shape: (3, 13, T)

                    filenames.append(filename)
                    labels.append(meta_dict['audio/' + filename])
                    features.append(feature_tensor)

            except Exception as e:
                print(f"Erro em {name}: {e}")

# Criação do DataFrame
df = pd.DataFrame({
    'filename': filenames,
    'label': labels,
    'features': features
})

df.to_pickle("mfcc_features_cnn.pkl")
