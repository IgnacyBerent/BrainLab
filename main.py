import prsa
import prsa
import prep
import matplotlib.pyplot as plt


def main():
    df = prep.read_data('r2.csv')
    df = prep.find_longest_segment(df)
    prsa_result = prsa.calc_optimized(df, 300)
    prsa.plot(prsa_result)


if __name__ == '__main__':
    main()