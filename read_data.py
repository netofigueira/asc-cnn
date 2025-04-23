import os
import librosa
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# Caminhos
zip_dir =  r"C:\Users\netog\Downloads\3670185"  # <- ajuste aqui
meta_file = r"C:\Users\netog\Downloads\3670185\meta.csv"

# Leitura do arquivo de metadados
meta = pd.read_csv(meta_file, sep='\t')
meta_dict = dict(zip(meta['filename'], meta['scene_label']))
print(meta_dict)
zip_files = sorted([f for f in os.listdir(zip_dir) if f.endswith('.zip') and'audio' in f])
# Parâmetros MFCC
sr = 22050
n_mfcc = 13
# Listas para DataFrame final
filenames, labels, mfcc_features = [], [], []


# Processa cada zip
for zip_name in tqdm(zip_files):
    with zipfile.ZipFile(os.path.join(zip_dir, zip_name), 'r') as z:
        for name in z.namelist():
            filename = name.split('/')[-1]
            print(filename)
            if not filename.endswith('.wav'):
                continue
            try:
                with z.open(name) as wav_file:
                    y, _ = sf.read(wav_file)
                    if y.ndim > 1:
                        y = np.mean(y, axis=1)

                    y = librosa.resample(y, orig_sr=48000, target_sr=sr)

                    # MFCC + delta + delta2
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    delta = librosa.feature.delta(mfcc)
                    delta2 = librosa.feature.delta(mfcc, order=2)

                    feature_vector = np.concatenate([
                        np.mean(mfcc, axis=1),
                        np.mean(delta, axis=1),
                        np.mean(delta2, axis=1)
                    ])

                    filenames.append(filename)
                    labels.append(meta_dict['audio/' + filename])
                    mfcc_features.append(feature_vector)
            except Exception as e:
                print(f"Erro em {name}: {e}")

# Criação do DataFrame
df = pd.DataFrame({
    'filename': filenames,
    'label': labels,
    'features': mfcc_features
})

df.to_pickle("mfcc_features.pkl")
df.to_csv("mfcc_features.csv")
