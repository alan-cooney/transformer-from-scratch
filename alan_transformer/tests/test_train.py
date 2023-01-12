import torch
from alan_transformer.train import one_hot_encode_inputs

class TestOneHotEncodeInputs:
    def test_simple_batch(self):
        example_batch = [
            torch.tensor([1, 3]),
            torch.tensor([2, 2])
        ]
        
        expected_res = torch.tensor([
          [[0., 1., 0., 0.], [0., 0., 0., 1.]],
          [[0., 0., 1., 0.], [0., 0., 1., 0.]]
        ])
        
        res = one_hot_encode_inputs(example_batch, 4)
        assert torch.allclose(res, expected_res)
        
    def test_shape_correct(self):
        batch_size = 2
        pos_size = 3
        vocab_size = 5
        example_batch = [torch.tensor([1, 4, 3]) for _batch in range(batch_size)]
        
        res = one_hot_encode_inputs(example_batch, vocab_size)
        assert res.size() == (batch_size, pos_size, vocab_size)
        
        
        
        
        
        
        