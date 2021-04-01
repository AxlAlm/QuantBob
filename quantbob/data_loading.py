




class DataLoader:



    def __init__(self):
        pass

    training_data = read_csv("numerai_training_data.csv")
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament_data = read_csv("numerai_tournament_data.csv")