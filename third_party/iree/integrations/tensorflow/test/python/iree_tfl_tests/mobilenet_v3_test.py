# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import absl.testing
import numpy
from . import test_util

model_path = "https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1?lite-format=tflite"


class MobilenetV3Test(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(MobilenetV3Test, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(MobilenetV3Test, self).compare_results(
            iree_results, tflite_results, details
        )
        self.assertTrue(
            numpy.isclose(iree_results[0], tflite_results[0], atol=1e-4).all()
        )

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
