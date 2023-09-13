import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def calc(df: pd.DataFrame, L: int) -> np.array:
    """
    Oblicza prsa według punktu 2 z https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param df: dane sygnału abp[mmHg]
    :param L: okno sygnału, gdzie 2L > najwolniejszej oscylacji
    :return: wartości prsa
    """

    x = df['abp[mmHg]'].to_numpy()
    indexs = np.arange(len(x))

    # wybieram punkty kotwicze według zależności X_i > X_i-1,
    # nie zaliczam punktów kotwiczych dla których okno wyszłoby poza zakres wartości
    anchor_points = np.array([])
    for i in indexs[L:-L]:
        if x[i] > x[i-1]:
            anchor_points = np.append(anchor_points, i)

    M = len(anchor_points)

    # obliczam średnie od k
    X_iv = np.array([])
    X_k = np.array([])
    for k in range(-L, L):
        for v in range(M):
            X_iv = np.append(X_iv, x[v+k])
        X_k = np.append(X_k, X_iv.mean())

    return X_k



def calc_optimized(df: pd.DataFrame, L: int) -> np.array:
    """
    Oblicza prsa według punktu 2 z https://www.sciencedirect.com/science/article/pii/S037843710501006X
    :param df: dane sygnału abp[mmHg]
    :param L: okno sygnału, gdzie 2L > najwolniejszej oscylacji
    :return: wartości prsa
    """

    x = df['abp[mmHg]'].to_numpy()

    # wybieram punkty kotwicze według zależności X_i > X_i-1,
    # nie zaliczam punktów kotwiczych dla których okno wyszłoby poza zakres wartości
    anchor_points = np.where(x[1:-L] > x[:-L - 1])[0] + 1

    # obliczam średnie od k
    X_k = []
    for k in range(-L, L):
        X_iv = x[anchor_points + k]
        X_k.append(np.mean(X_iv))

    return np.array(X_k)


def plot(prsa_values: np.array):

    x = np.arange(len(prsa_values))

    # Ustawienie ośi y jako wartości z prsa_values
    y = prsa_values

    # Usunięcie wartości NaN i Inf przed obliczaniem min i max
    valid_values = prsa_values[~np.isnan(prsa_values) & ~np.isinf(prsa_values)]

    # Jeśli wszystkie wartości są NaN lub Inf, zwróć bez rysowania wykresu
    if len(valid_values) == 0:
        print("Brak poprawnych wartości do wyrysowania wykresu.")
        return

    # Ustal długość zapasu na osi y (10% z dołu i góry)
    y_min = np.min(valid_values) * 0.9
    y_max = np.max(valid_values) * 1.1

    # Ustawienie limitów osi x i y
    plt.xlim(0, len(prsa_values))
    plt.ylim(y_min, y_max)

    # Tworzenie wykresu scatter
    plt.scatter(x, y)

    # Dodanie tytułu wykresu
    plt.title("Phase Rectified Signal Average")

    # Etykiety osi
    plt.xlabel("Index k")
    plt.ylabel("X(k) [mm[Hg]]")

    # Wyświetlenie wykresu
    plt.show()






