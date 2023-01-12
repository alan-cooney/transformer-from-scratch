from alan_transformer.examples.shakespeare import (create_dataloader,
                                                   create_dataset,
                                                   create_tokenizer,
                                                   tokenize_prompt)


class TestTokenizePrompt:
    def test_input_ids_size(self):
        tokenizer = create_tokenizer()
        text = ["This is a text string"]
        res = tokenize_prompt(text, tokenizer)
        assert res["input_ids"].size() == (1, 1024)

class TestCreateDataset:
    def test_first_input_ids_size(self):
        dataset = create_dataset(
            # load_if_exists=False
            )
        first_example = dataset["train"][0]
        assert first_example["input_ids"].size() == (1024, )
        
class TestCreateDataloader:
    def test_first_batch_size(self):
        dataset = create_dataset()
        batch_size=8
        dataloader = create_dataloader(dataset, batch_size=batch_size)
        first_batch_item = next(iter(dataloader))
        assert first_batch_item["input_ids"].size() == (batch_size, 1024)