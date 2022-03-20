


from quantbob.setup_pl import setup_study
from quantbob.setup_pl import setup_study


if __name__ == '__main__':
    
    # Initialparse some args
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--tune", default=False, action="store_true")
    args = parser.parse_args()

    # set up dataset
    dataset =  NumerAIDataset(debug = self._debug)


    setup(
        commet_logger = comet_logger
        )