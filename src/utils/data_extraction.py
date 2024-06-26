import numpy as np
import pandas as pd
from tqdm import tqdm

def add_acceleration_columns(df):
    acc_x_list = []
    acc_y_list = []

    # Usamos tqdm para mostrar la barra de progreso
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculando aceleraciones'):
        x = np.array(row['cop_x'])
        y = np.array(row['cop_y'])
        
        # Calcular las diferencias una vez y dividir por el cuadrado del intervalo de tiempo
        acc_x = np.diff(x) / (0.04)
        acc_y = np.diff(y) / (0.04)
        
        # Agregar un valor inicial de 0 para mantener la longitud de la serie de tiempo
        acc_x = np.insert(acc_x, 0, 0)
        acc_y = np.insert(acc_y, 0, 0)
        
        # Agregar las aceleraciones a las listas
        acc_x_list.append(acc_x)
        acc_y_list.append(acc_y)

    # Asignar las listas de aceleraciones al DataFrame
    df['acc_x'] = acc_x_list
    df['acc_y'] = acc_y_list

def add_path_column(df):
    path_list = []

    # Usamos tqdm para mostrar la barra de progreso
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculando longitud del camino'):
        x = np.array(row['cop_x'])
        y = np.array(row['cop_y'])
        path_length = sum(np.sqrt(x ** 2 + y ** 2))
        path_list.append(path_length)

    df['path'] = path_list

def add_rms_columns(df):
    rms_x_list = []
    rms_y_list = []

    # Usamos tqdm para mostrar la barra de progreso
    for index, row in tqdm(df.iterrows(), total=len(df), desc='Calculando RMS'):
        x = np.array(row['acc_x'])
        y = np.array(row['acc_y'])
        rms_x = np.sqrt(np.mean(x ** 2))
        rms_y = np.sqrt(np.mean(y ** 2))
        rms_x_list.append(rms_x)
        rms_y_list.append(rms_y)

    df['rms_acc_x'] = rms_x_list
    df['rms_acc_y'] = rms_y_list

def add_sampen_columns(df, m=2, r=0.2):
    """
    Calcula y añade la entropía muestral (Sample Entropy) para las series de tiempo X e Y en el DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame que contiene las columnas 'cop_x' y 'cop_y'.
        m (int): Longitud de la subserie.
        r (float): Umbral de similitud.

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas 'samp_en_x' y 'samp_en_y'.
    """
    def get_sampen_series(series, m, r):
        """
        Calcula la entropía muestral (Sample Entropy) para una serie de tiempo.

        Parameters:
            series (list or numpy array): Serie de tiempo.
            m (int): Longitud de la subserie.
            r (float): Umbral de similitud.

        Returns:
            float: Entropía muestral.
        """
        N = len(series)
        B = 0.0
        A = 0.0
        # Divide la serie temporal y guarda todas las plantillas de longitud m.
        xmi = np.array([series[i : i + m] for i in range(N - m)])
        xmj = np.array([series[i : i + m] for i in range(N - m + 1)])
        # Guarda todas las coincidencias menos la coincidencia propia y calcula B.
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
        # Similar para calcular A.
        m += 1
        xm = np.array([series[i : i + m] for i in range(N - m + 1)])
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
        # Devuelve la entropía muestral.
        return -np.log(A / B)

    # Usamos tqdm para mostrar la barra de progreso
    for col in ['cop_x', 'cop_y']:
        desc = f'Calculando Sample Entropy para {col}'
        df[f'samp_en_{col[-1]}'] = df[col].apply(lambda series: get_sampen_series(series, m, r))

def add_f80_columns(df):
    """
    Calcula y añade el f80 para las series de tiempo X e Y en el DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame que contiene las columnas 'cop_x' y 'cop_y'.

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas 'f80_x' y 'f80_y'.
    """
    def get_fft(serie):
        f = np.linspace(0.0, 1.0 / (2.0 * (1 / 25)), 500 // 2)
        power = np.abs(np.fft.fft(serie))[0:500 // 2]
        return f[f <= 4], power[f <= 4]

    def get_f80(f, power, min_val, max_val):
        return np.sum(power[(f >= min_val) & (f < max_val)]) * 4 / 5    

    def get_f80_on_column(serie):
        # Obtener las frecuencias y potencias utilizando get_fft
        f, power = get_fft(serie)
        
        # Calcular el f80 utilizando las frecuencias y potencias
        f80 = get_f80(f, power, 0, 4)
        
        return f80

    # Usamos tqdm para mostrar la barra de progreso
    for col in ['cop_x', 'cop_y']:
        desc = f'Calculando F80 para {col}'
        df[f'f80_{col[-1]}'] = df[col].apply(get_f80_on_column)

def add_frequency_features(df):
    """
    Calcula y añade las frecuencias medias para las series de tiempo X e Y en el DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame que contiene las columnas 'cop_x' y 'cop_y'.

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas de frecuencias medias.
    """
    def get_fft(serie):
        f = np.linspace(0.0, 1.0 / (2.0 * (1 / 25)), 500 // 2)
        power = np.abs(np.fft.fft(serie))[0:500 // 2]
        return f[f <= 4], power[f <= 4]

    def get_mean_freq(f, power, min_freq, max_freq):
        mean = np.mean(power[(f >= min_freq) & (f < max_freq)])
        return mean

    def add_frequency_features_to_df(df, prefix, min_freq, max_freq):
        # Aplicar la función de frecuencia media para las series de tiempo X e Y
        df[f'{prefix}_x'] = df['cop_x'].apply(lambda x: get_mean_freq(*get_fft(x), min_freq, max_freq))
        df[f'{prefix}_y'] = df['cop_y'].apply(lambda y: get_mean_freq(*get_fft(y), min_freq, max_freq))

    # Definir los rangos de frecuencia para cada característica
    frequency_ranges = {
        'mf_lf': (0, 0.5),
        'mf_mf': (0.5, 2),
        'mf_hf': (2, 4)
    }

    # Usamos tqdm para mostrar la barra de progreso
    for feature, (min_freq, max_freq) in tqdm(frequency_ranges.items(), desc='Calculando frecuencias medias'):
        add_frequency_features_to_df(df, feature, min_freq, max_freq)

def get_features(data):
    """
    Aplica múltiples métodos de extracción de características a un DataFrame dado.

    Parameters:
    - data (DataFrame): El DataFrame al que se aplicarán los métodos de extracción.

    Returns:
    - DataFrame: El DataFrame modificado con las nuevas características agregadas.
    """
    # Aplicar cada método de extracción con tqdm para mostrar la barra de progreso
    add_acceleration_columns(data)
    add_rms_columns(data)
    add_path_column(data)
    add_sampen_columns(data)
    add_f80_columns(data)
    add_frequency_features(data)

    return data