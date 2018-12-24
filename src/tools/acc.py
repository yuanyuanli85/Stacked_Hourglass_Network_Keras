import argparse
import matplotlib.pyplot as plt
import os


def get_number(valfile):
    with open(valfile) as xfile:
        lines = xfile.readlines()

    scores = []
    for _line in lines:
        _line = _line.strip()
        val_score = float(_line.split(':')[-1])
        scores.append(val_score)

    return scores


def main_plot(vallst):
    for valfile in vallst:
        valname = os.path.basename(os.path.dirname(valfile))
        score = get_number(valfile)
        plt.plot(score, label=valname)

    plt.title('MPII val score')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_file", '--list', action='append', help='val file with validation score')

    args = parser.parse_args()
    main_plot(args.val_file)
