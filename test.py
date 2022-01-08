# from quantbob.numerai_dataset import NumerAIDataset

# dataset = NumerAIDataset()
# cv = dataset.create_cvs(n_folds = 10)

import pyarrow.parquet import ParquetFile


F = ParquetFile("/tmp/numerai_data/numerai_validation_data.parquet")