import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np

def scaling(X, num_samples=1, scale_factors=[0.7, 0.8, 0.9, 1.1, 1.2, 1.3]):
    """
    Augmenta un conjunto de series de tiempo utilizando escalado aleatorio.

    Parameters:
    X (list): Conjunto de series de tiempo originales.
    num_samples (int): Número de muestras de series de tiempo escaladas para generar.
                       Por defecto es 1.
    scale_factors (list): Lista de factores de escala para seleccionar al azar.
                         Por defecto son [0.7, 0.8, 0.9, 1.1, 1.2, 1.3].

    Returns:
    augmented_data (list): Conjunto de series de tiempo aumentadas.
    """
    X = np.array([np.array(series) for series in X])

    augmented_data = []
    
    # Calculamos cuántas series de tiempo escaladas necesitamos generar
    current_samples = len(X)
    total_samples_needed = num_samples - current_samples
    
    if total_samples_needed > 0:
        # Generamos series de tiempo escaladas adicionales hasta alcanzar el número deseado
        while total_samples_needed > 0:
            series_idx = np.random.randint(0, current_samples)  # Seleccionamos una serie al azar
            scale_factor = np.random.choice(scale_factors)     # Seleccionamos un factor de escala al azar
            augmented_series = X[series_idx] * scale_factor    # Escalamos la serie seleccionada
            augmented_data.append(augmented_series)
            total_samples_needed -= 1

    return augmented_data


import numpy as np

def shuffle_time_slices(time_series, slice_size):
    """
    Shuffle different time slices of the provided array.

    Parameters:
    time_series (array-like): An array containing time-series data.
    slice_size (int): The size of each time slice that will be shuffled.

    Returns:
    shuffled_data (array-like): The array with shuffled time slices.
    """
    time_series = np.array(time_series)

    if slice_size <= 0 or slice_size > len(time_series):
        raise ValueError("Slice size must be within the range 1 to len(data)")

    num_slices = len(time_series) // slice_size

    slices = [time_series[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]

    np.random.shuffle(slices)

    shuffled_data = np.concatenate(slices)

    remainder = len(time_series) % slice_size
    if remainder > 0:
        remainder_data = time_series[-remainder:]
        shuffled_data = np.concatenate([shuffled_data, remainder_data])

    return shuffled_data

def random_shuffling(time_series_set, num_samples, slice_size=100):
    """
    Augment time series data by applying random shuffling using the shuffle_time_slices function.

    Parameters:
    time_series_set (array-like): Set of time series data.
    num_samples (int): Number of artificial samples to generate.
    slice_size (int): The size of each time slice that will be shuffled.

    Returns:
    augmented_data (array-like): Augmented time series data.
    """
    time_series_set = np.array([np.array(series) for series in time_series_set])

    augmented_data = []

    # Calcular el número mínimo de muestras requeridas para alcanzar num_samples
    min_samples_needed = max(num_samples, 1600)

    # Generar suficientes muestras para alcanzar o superar min_samples_needed
    for _ in range(min_samples_needed):
        # Selección aleatoria de una serie de tiempo del conjunto
        idx = np.random.randint(len(time_series_set))
        time_series = time_series_set[idx]
        
        # Aplicar el shuffle de las slices
        shuffled_slices = shuffle_time_slices(time_series, slice_size=slice_size)
        
        augmented_data.append(shuffled_slices)

    return augmented_data

def detect_outliers(series, threshold=3):
    """
    Detecta outliers en una serie de tiempo utilizando la desviación estándar.

    Parameters:
    series (np.array): Serie de tiempo.
    threshold (float): Umbral multiplicativo para la desviación estándar.

    Returns:
    boolean: True si hay outliers, False si no.
    """
    std = np.std(series)
    return any(abs(series - np.mean(series)) > threshold * std)

def window_warping(X, num_samples=1600, window_size_ratio=0.5, scale_factors=[0.9, 1.1], outlier_threshold=3):
    """
    Augmenta un conjunto de series de tiempo utilizando window warping con ajustes,
    omitiendo las series que contienen outliers.

    Parameters:
    X (list): Conjunto de series de tiempo originales.
    num_samples (int): Número aproximado de muestras artificiales a generar.
    window_size_ratio (float): Ratio del tamaño de la ventana respecto a la longitud total de la serie de tiempo.
    scale_factors (list): Lista de factores de escala para seleccionar al azar.
    outlier_threshold (float): Umbral para detectar outliers.

    Returns:
    augmented_data (list): Conjunto de series de tiempo aumentadas.
    """
    augmented_data = []
    
    # Filtrar series de tiempo que contienen outliers
    X_filtered = [series for series in X if not detect_outliers(series, outlier_threshold)]
    
    # Calcula cuántas veces necesitas repetir el conjunto original para alcanzar num_samples
    num_repeats = int(np.ceil(num_samples / len(X_filtered)))
    
    for _ in range(num_repeats):
        for series in X_filtered:
            series_length = len(series)
            window_size = int(series_length * window_size_ratio)
            window_size = max(window_size, 1)
            
            start_idx = np.random.randint(0, series_length - window_size + 1)
            scale_factor = np.random.choice(scale_factors)
            
            window = series[start_idx:start_idx + window_size]
            warped_window = np.interp(
                np.linspace(0, window_size, int(window_size * scale_factor)),
                np.arange(window_size),
                window
            )
            
            augmented_series = np.concatenate([
                series[:start_idx],
                warped_window,
                series[start_idx + window_size:]
            ])
            
            if len(augmented_series) > series_length:
                augmented_series = augmented_series[:series_length]
            elif len(augmented_series) < series_length:
                augmented_series = np.pad(augmented_series, (0, series_length - len(augmented_series)), 'constant')
            
            augmented_data.append(augmented_series)
            
            # Si hemos alcanzado el número deseado de muestras, salir del bucle
            if len(augmented_data) >= num_samples:
                break
        
        # Si hemos alcanzado el número deseado de muestras, salir del bucle externo
        if len(augmented_data) >= num_samples:
            break
    
    return augmented_data


import numpy as np
import numpy as np

def add_gaussian_noise(time_series, sigma_range=(0.02, 0.04)):
    """
    Añade ruido gaussiano a una serie de tiempo.

    Parameters:
    time_series (list or np.ndarray): La serie de tiempo original.
    sigma_range (tuple): Rango de desviación estándar del ruido gaussiano.
                        Por defecto es (0.02, 0.04).

    Returns:
    np.ndarray: La serie de tiempo con ruido gaussiano añadido.
    """
    # Asegurarse de que la serie de tiempo sea un array numpy
    time_series = np.array(time_series)
    
    # Generar sigma aleatorio dentro del rango especificado
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    
    return time_series + np.random.normal(loc=0., scale=sigma, size=time_series.shape)

def augment_time_series_with_noise(X, num_samples=1):
    """
    Augmenta un conjunto de series de tiempo añadiendo ruido gaussiano múltiples veces.

    Parameters:
    X (list of lists or np.ndarray): Conjunto de series de tiempo originales.
    num_samples (int): Número total de muestras ruidosas que se quiere generar.
                      Por defecto es 1.

    Returns:
    augmented_data (list): Conjunto de series de tiempo con ruido gaussiano añadido.
    """
    # Calcular cuántas muestras por serie son necesarias para alcanzar num_samples
    samples_per_series = int(np.ceil(num_samples / len(X)))    
    # Inicializar lista para almacenar las series de tiempo aumentadas
    augmented_data = []
    
    # Iterar sobre cada serie de tiempo en el conjunto original
    for series in X:
        # Generar muestras ruidosas para esta serie de tiempo
        for _ in range(samples_per_series):
            # Añadir ruido gaussiano a la serie de tiempo
            noisy_series = add_gaussian_noise(series)
            
            # Agregar la serie de tiempo con ruido a la lista de muestras
            augmented_data.append(noisy_series)
    
    return augmented_data


def flip_time_series_set(X):
    """
    Invierte un conjunto de series de tiempo.

    Parameters:
    X (list): Conjunto de series de tiempo originales.

    Returns:
    flipped_data (list): Conjunto de series de tiempo invertidas.
    """
    # Inicializar lista para almacenar las series de tiempo invertidas
    flipped_data = []
    
    # Iterar sobre cada serie de tiempo en el conjunto original
    for series in X:
        # Invertir la serie de tiempo
        flipped_series = np.flip(series)
        
        # Agregar la serie de tiempo invertida a la lista
        flipped_data.append(flipped_series)
    
    return flipped_data