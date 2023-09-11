import numpy as np
import pandas as pd
import pandas

def prsa(df: pd.DataFrame, window_size):

    signal_data = df['abp[mmHg]'].to_numpy()
    phase_data = df['Phases'].to_numpy()

    # Inicjalizacja pustej listy na wynikowy sygnał PRSA
    prsa_signal = []

    # Przetwarzanie sygnału dla każdego okna czasowego
    for i in range(len(signal_data) - window_size + 1):
        # Wybieranie danej fazowej w bieżącym oknie czasowym
        phase_window = phase_data[i:i+window_size]

        # Obliczanie średniej fazowej
        mean_phase = np.mean(phase_window)

        # Wyznaczanie amplitudy dla każdej próbki w oknie czasowym
        amplitude_window = [signal_data[i+j] * np.cos(phase - mean_phase)
                            for j, phase in enumerate(phase_window)]

        # Obliczanie średniej amplitudy
        mean_amplitude = np.mean(amplitude_window)

        # Dodawanie średniej amplitudy do wynikowego sygnału PRSA
        prsa_signal.append(mean_amplitude)
    return prsa_signal


def calculate_phase(icp_data):
    # Wykonaj dyskretną transformację Fouriera (FFT) na sygnale ICP.
    fft_result = np.fft.fft(icp_data)

    # Znajdź indeks częstotliwości o największej amplitudzie (to będzie częstotliwość dominująca).
    dominant_frequency_index = np.argmax(np.abs(fft_result))

    # Oblicz fazę na podstawie indeksu częstotliwości dominującej.
    phase = np.angle(fft_result[dominant_frequency_index])

    return phase

def make_phase_column(df):
    # Pobierz dane ICP z kolumny "abp[mmHg]" jako numpy array.
    icp_data = df["abp[mmHg]"].to_numpy()

    # Oblicz fazę sygnału ICP.
    phase = calculate_phase(icp_data)

    # Dodaj nową kolumnę "Phases" do DataFrame z obliczonymi fazami.
    df["Phases"] = phase
    return df








