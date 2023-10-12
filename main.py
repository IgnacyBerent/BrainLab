import prsa
import prep


def main():
    df = prep.read_data('data/r2.csv')
    df = prep.find_longest_segments(df)
    prsa_result = prsa.calculate(df[0], 300)
    prsa.plot(prsa_result)


if __name__ == '__main__':
    main()
