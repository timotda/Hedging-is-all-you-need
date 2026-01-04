import numpy as np


def load_AAPL_data(data_path, seq_len, dim):
    data = np.load(data_path)
    return chop_into_windows(data, seq_len, stride=1)

def load_stocks_data(data_path, seq_len, dim):
    data = np.load(data_path)
    return chop_into_windows(data, seq_len, stride=1)

def minmax_scale_features(data):
    """
    Scales the features of the data to the range [0, 1] using min-max scaling.

    Args:
        data (np.ndarray): The data to be scaled.

    Returns:
        tuple: A tuple containing the scaled data, the minimum values, and the maximum values for each feature.
    """
    data_min = data.min(axis=(0, 1), keepdims=True)
    data_max = data.max(axis=(0, 1), keepdims=True)
    data = (data - data_min) / (data_max - data_min + 1e-6)
    return data, data_min, data_max


def reverse_minmax_scaler(real_path_folder, name, data):
    """
    Reverses the min-max scaling of the data using the scaling factors from the real data.

    Args:
        real_path_folder (str): The folder containing the real data.
        name (str): Experiment name.
        data (np.ndarray): The data to be rescaled.

    Returns:
        np.ndarray: The rescaled data.
    """
    real_paths = np.load(f"{real_path_folder}{name}.npy")
    real_paths_scaled, data_min, data_max = minmax_scale_features(real_paths)
    return data * (data_max - data_min + 1e-6) + data_min


def clip_quantiles(data, q1, q2):
    data = np.clip(data, np.quantile(data, q1, axis=0), np.quantile(data, q2, axis=0))
    return data


def clip_outliers(data, n_std=4):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return np.clip(data, mean - n_std * std, mean + n_std * std)


def chop_into_windows(ts, window_size, stride):
    """
    Chops a time series into overlapping windows.

    Parameters:
    ts (numpy.ndarray): A numpy array of shape (length, dim) representing the time series.
    window_size (int): The size of each window.
    stride (int): The number of steps to move the window forward.

    Returns:
    numpy.ndarray: A numpy array of shape (num_windows, window_size, dim) containing the windows.
    """
    windows = []
    for i in range(0, len(ts) - window_size, stride):
        windows.append(np.array(ts[i : i + window_size, :]))
    windows = np.array(windows)
    return windows
