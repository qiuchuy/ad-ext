import ailang as al
import csv
import time

from ailang.nn import resnet

batch_vec = [1, 2, 4]  # , 8, 16, 32, 64, 128]
coco_shape = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640]
num_trials = 100  # Number of runs to average the inference time

# Open CSV file for writing
with open("inference_time.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Batch Size", "Input Shape", "Average Inference Time (ms)"])

    for b in batch_vec:
        for hw in coco_shape:
            avg_time = al.nn.measure_inference_time(
                resnet, al.randn([b, 3, hw, hw]), num_trials
            )
            print(
                f"Batch size: {b}, Input shape: ({hw}, {hw}), Average inference time: {avg_time:.2f} ms"
            )
