import numpy as np
import argparse

from utils import Perceptron, compute_accuracy_score, compute_cross_val_score


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cvscore", help="The number of folds in order to perform cross validation", type=int, metavar="")
parser.add_argument("-s", "--size", help="Matrix dimensions (sxs)", type=int, metavar="")
args = parser.parse_args()


def main():

    if args.cvscore and args.size:
        X = np.arange(args.size**2).reshape((args.size, args.size))
        y = np.arange(args.size)

        compute_cross_val_score(X, y, cv=args.cvscore)


if __name__ == "__main__":
    main()
