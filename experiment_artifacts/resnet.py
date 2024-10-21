import ailang as al
import csv
import time

from ailang.nn import resnet50

batch_vec = [1, 2, 4]
coco_shape = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640]
num_trials = 100  # Number of runs to average the inference time

# Open CSV file for writing
with open("/workspace/inference_time.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    for b in batch_vec:
        for hw in coco_shape:
            avg_time = al.nn.measure_inference_time(
                resnet50, al.randn([b, 3, hw, hw]), num_trials
            )
            print(
                f"Batch size: {b}, Input shape: ({hw}, {hw}), Average inference time: {avg_time:.2f} ms"
            )
            writer.writerow([b, f'{hw}x{hw}', f'{avg_time:.2f}'])
