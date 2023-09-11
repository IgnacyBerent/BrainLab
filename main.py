from prsa import *
from prep import *
import matplotlib.pyplot as plt

def main():
    data = read_data('r2.csv')
    data = make_phase_column(data)
    prsa_signal = prsa(data, 1)


if __name__ == '__main__':
    main()