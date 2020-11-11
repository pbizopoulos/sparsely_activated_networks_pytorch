import sparsely_activated_networks_pytorch as san
import torch
import unittest


class TestSuite(unittest.TestCase):
    def test_san1d(self):
        inputs = torch.randn(2, 1, 10)
        sparse_activation_list = [torch.nn.ReLU(), torch.nn.ReLU()]
        kernel_size_list = [3, 3]
        model = san.SAN1d(sparse_activation_list, kernel_size_list)
        outputs = model(inputs)

    def test_san2d(self):
        inputs = torch.randn(2, 1, 10, 10)
        sparse_activation_list = [torch.nn.ReLU(), torch.nn.ReLU()]
        kernel_size_list = [3, 3]
        model = san.SAN2d(sparse_activation_list, kernel_size_list)
        outputs = model(inputs)


if __name__ == '__main__':
    unittest.main()
