
from argparse import ArgumentParser
from quantbob import QuantBob




if __name__ == '__main__':
    
    # Initialparse some args
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    #parser.add_argument("--upload", default=False, action="store_true")
    #paser.add_argument("--API_KEY")
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    args = parser.parse_args()
    
    # start quantbob
    QuantBob(config_path = args.config, debug = args.debug)