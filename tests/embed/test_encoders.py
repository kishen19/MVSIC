from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from embed.encoders import ColBERTEncoder


class ColBERTWrapperTest(unittest.TestCase):
    def test_ragged_documents_move_to_cpu_individually(self):
        arrays = [np.array([[3., 4.], [0., 0.]]), np.array([[0., 2.]])]
        tensors = []
        for array in arrays:
            tensor = Mock()
            tensor.detach.return_value = tensor
            tensor.float.return_value = tensor
            tensor.cpu.return_value = tensor
            tensor.numpy.return_value = array
            tensors.append(tensor)

        def ragged_documents(values, **kwargs):
            self.assertEqual(values, ["first", "second"])
            self.assertFalse(kwargs["keep_dims"])
            # ColBERT releases that call .cpu() on the ragged list fail here.
            self.assertFalse(kwargs["to_cpu"])
            return tensors

        encoder = ColBERTEncoder.__new__(ColBERTEncoder)
        encoder._torch = SimpleNamespace(no_grad=nullcontext)
        encoder._model = SimpleNamespace(docFromText=ragged_documents)
        encoder.zero_atol = 1e-9
        result = encoder.encode_documents(["first", "second"])
        np.testing.assert_allclose(result[0], [[.6, .8]], rtol=1e-6)
        np.testing.assert_array_equal(result[1], [[0., 1.]])
        for tensor in tensors:
            tensor.cpu.assert_called_once()
        self.assertTrue(all(a.dtype == np.float32 and a.flags.c_contiguous for a in result))


if __name__ == "__main__":
    unittest.main()
