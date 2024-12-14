import ailang as al
import numpy as np
from ailang import lstm

kInputSize = 256
kHiddenSize = 256
kCellNumber = 10
kLstmTimestep = 100

model = lstm(kInputSize, kHiddenSize, kCellNumber, kLstmTimestep)

min, max, mean = al.nn.measure_inference_time(model, al.from_numpy(np.random.randn(kLstmTimestep, kCellNumber, kInputSize).astype(np.float32), device="gpu"), show_min_max=True)
print("Average time(us):")
print(mean)


