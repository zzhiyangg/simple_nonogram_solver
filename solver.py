from nonogram import *

def main():
    puzzle_file = 'nonogram.csv'
    nng = nonogram(puzzle_file)
    nng.solve()

if __name__ == '__main__':
    main()