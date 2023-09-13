import prsa
import prsa
import prep
import matplotlib.pyplot as plt


def main():
    df = prep.read_data('r2.csv')
    prsa.plot(prsa.calc_optimized(df, 300))


if __name__ == '__main__':
    main()