import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def main(args):
    result = pickle.load(open(args.results_path, "rb"))

    print(result.keys())

    plt.subplot(1, 2, 1)
    plt.plot(result['loss'], label='loss')
    plt.plot(result['val_loss'], label='val_loss')
    plt.ylim([0, 5])
    plt.legend()

    if 'acc' in result.keys():
        metrics = 'acc'
        y_range = [0.5, 1]
    elif 'mae' in result.keys():
        metrics = 'mae'
        y_range = [0, 5]
    else:
        print("Unknown metrics")
        raise NotImplementedError
    
    plt.subplot(1, 2, 2)
    plt.plot(result[metrics], label=metrics)
    plt.plot(result[f'val_{metrics}'], label=f'val_{metrics}')
    plt.ylim(y_range)
    plt.legend()

    plt.show()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--results_path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
