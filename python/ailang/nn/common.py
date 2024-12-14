import ailang as al
import torch

def measure_inference_time(model, input_tensor, show_min_max=False, runs=1):
    """
    Measure the average inference time of a model on CUDA.

    Args:
        model (nn.Module): The model to measure.
        input_tensor (Tensor): The input tensor for inference.
        runs (int): Number of runs to average the inference time over.

    Returns:
        float: Average inference time in milliseconds.
    """
    if model == "lstm":
        import subprocess
        import re

        def read_last_line(file_path):
            with open(file_path, 'r') as file:
                # Read the last line
                last_line = file.readlines()[-1]
                print(last_line.strip())
            
                # Extract min, max, mean values
                match = re.search(r'\[min, max, mean\] = \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]', last_line)
                if match:
                    min_val = float(match.group(1))
                    max_val = float(match.group(2))
                    mean_val = float(match.group(3))
                    return min_val, max_val, mean_val
                else:
                    raise ValueError("The expected pattern was not found in the last line.")

        print("Start fused kernel kernel optimized by AILang...")
        subprocess.run("python /workspace/extension-cpp/test/test_extension.py > /workspace/extension-cpp/results/fused_lstm.log", shell=True, check=True)
        file_path = '/workspace/extension-cpp/results/fused_lstm.log' 
        return read_last_line(file_path)

    # Create CUDA events to measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch_input_tensor = torch.from_numpy(al.to_numpy(input_tensor)).to(dtype=torch.half, device="cuda")

    # Run inference multiple times and measure the time
    total_time = 0
    for _ in range(runs):
        # torch_input_tensor = torch.from_numpy(al.to_numpy(input_tensor), dtype=torch.half, device="cuda")

        # Record the start time
        start_event.record()

        # Perform inference
        with torch.no_grad():
            model(torch_input_tensor)

        # Record the end time
        end_event.record()

        # Wait for the events to be completed
        torch.cuda.synchronize()

        # Calculate elapsed time for this run
        total_time += start_event.elapsed_time(end_event)

    # Calculate average time
    avg_time = total_time / runs
    return avg_time


