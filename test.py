from quantbob.numerai_dataset import NumerAIDataset

dataset = NumerAIDataset()
cv = dataset.create_cvs(n_folds = 10)
