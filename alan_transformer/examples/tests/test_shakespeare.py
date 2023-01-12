from alan_transformer.examples.shakespeare import create_dataset

class TestCreateDataset:
    def test_create_dataset(self):
        # It creates the dataset, with the first batch of the correct format
        res = create_dataset()