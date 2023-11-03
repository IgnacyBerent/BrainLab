import prsa
import prep


# Example usage of my libraries
def main():
    df = prep.read_data('data/r2.csv')
    df = prep.find_longest_segments(df)
    prsa_result = prsa.calculate(df[0][0], 300, "AC")
    prsa.plot(prsa_result)


if __name__ == '__main__':
    main()
