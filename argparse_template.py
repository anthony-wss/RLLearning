from argparse import ArgumentParser

def main(args):
    print(args.model)
    pass

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)