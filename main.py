import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import librosa
import librosa.display
import os
import zipfile
from urllib.request import urlretrieve


def spectrogram(samples, sample_rate, stride_ms = 10.0,
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples,
                                          shape = nshape, strides = nstrides)

    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]

    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2

    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale

    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    # ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:, :] + eps)
    return specgram, freqs

def pooling_audio(mat, ksize, method = 'max', pad = False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def process_audio(file_path, sr=None, pool_size=(4, 4)):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        spec, freqs = spectrogram(y, sr)
        if spec is None:
            return None, None, None, None, None
        spec_pooled = pooling_audio(spec, pool_size, method='max', pad=True)
        if spec_pooled is None:
            return None, None, None, None, None
        return os.path.basename(file_path), spec_pooled, y, sr, freqs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None, None

if __name__ == "__main__":
    # Download and extract dataset
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_file_path = "ESC-50-master.zip"
    download_path = "./ESC-50-master/"
    if not os.path.exists(download_path):
        try:
            urlretrieve(url, zip_file_path)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(zip_file_path)
        except Exception as e:
            print(f"Error downloading or extracting dataset: {e}")
            exit(1)

    # Load metadata and filter for 'dog' and 'chirping_birds'
    meta_file = os.path.join(download_path, "ESC-50-master/meta/esc50.csv")
    try:
        df_meta = pd.read_csv(meta_file)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        exit(1)

    categories = ['dog', 'chirping_birds']
    df_filtered = df_meta[df_meta['category'].isin(categories)]
    audio_files = [os.path.join(download_path, "ESC-50-master/audio", fname) for fname in df_filtered['filename']]
    audio_files = (
        df_filtered[df_filtered['category'] == 'dog']['filename'].tolist() +
        df_filtered[df_filtered['category'] == 'chirping_birds']['filename'].tolist()
    )
    audio_files = [os.path.join(download_path, "ESC-50-master/audio", fname) for fname in audio_files]
    n_signals = len(audio_files)

    # Processing audio files
    spectrograms = {}
    audio_data = {}
    freqs_list = {}
    true_labels = []
    for file_path in audio_files:
        file_name, spec_pooled, y, sr, freqs = process_audio(file_path)
        if file_name is None or spec_pooled is None:
            continue
        if spec_pooled.size == 0:
            print(f"Skipping {file_name}: Empty spectrogram")
            continue
        category = df_meta[df_meta['filename'] == file_name]['category'].values
        if len(category) == 0:
            print(f"Skipping {file_name}: No category found")
            continue
        category = category[0]
        spectrograms[file_name] = spec_pooled
        audio_data[file_name] = (y, sr)
        freqs_list[file_name] = freqs
        true_labels.append(category)
        print(f"Processed {file_name}: Spectrogram shape {spec_pooled.shape}, Category {category}, Audio length {len(y)} samples, Sample rate {sr} Hz")

    # Visualize spectrograms for one file per category
    for category in categories:
        matching_files = df_filtered[df_filtered['category'] == category]['filename']
        file_name = matching_files.iloc[0]
        if file_name in spectrograms:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(spectrograms[file_name], sr=audio_data[file_name][1], x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram for {file_name} ({category})')
            plt.show()

    # Flatten spectrograms for clustering
    spec_flattened = [spec.flatten() for spec in spectrograms.values()]
    spec_flattened = np.array(spec_flattened)

    # Spectral Clustering with kneighbors_graph
    n_clusters = 2
    adjacency_matrix = kneighbors_graph(spec_flattened, n_neighbors = 5, mode = 'connectivity', include_self = False)
    clustering = SpectralClustering(n_clusters = n_clusters, affinity = 'nearest_neighbors', random_state = 42)
    predicted_labels = clustering.fit_predict(adjacency_matrix)

    # Analyze clusters
    print("\nClustering labels for audio files:")
    for file, label, category in zip(spectrograms.keys(), predicted_labels, true_labels):
        print(f"{file}: Cluster {label}, Category {category}")
    
    true_label_map = {'dog': 0, 'chirping_birds': 1}
    true_labels_numeric = [true_label_map[category] for category in true_labels]

    # Use PCA to reduce dimensionality for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2, random_state = 42)
    spec_2d = pca.fit_transform(spec_flattened)

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(spec_2d[:, 0], spec_2d[:, 1], c = predicted_labels, cmap = 'viridis', s = 100)
    plt.colorbar(scatter, label='Cluster')

    for i, (file, category) in enumerate(zip(spectrograms.keys(), true_labels)):
        plt.annotate(category, (spec_2d[i, 0], spec_2d[i, 1]), fontsize = 8, alpha = 0.7)
    plt.title('Visualization of Audio Clusters with k-Neighbors Graph (Dog vs Chirping Birds)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    # Conclusion 
    print("""
    Висновок про значення застосування перетворення Фур'є:
    Перетворення Фур'є, використане у функції `spectrogram`, є ключовим для вилучення ознак з аудіоданих. 
    Воно дозволяє перейти від часового сигналу до частотного, виділяючи спектральні характеристики, 
    такі як домінуючі частоти та їх розподіл у часі. Собаки мають нижчі частоти та імпульсні характеристики, 
    тоді як птахи характеризуються вищими частотами. 

    Без перетворення Фур'є часові сигнали були б менш інформативними, ускладнюючи розрізнення звуків.
    """)
