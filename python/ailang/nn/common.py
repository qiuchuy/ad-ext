import ailang as al
import torch

def measure_inference_time(model, input_tensor, runs=1):
    """
    Measure the average inference time of a model on CUDA.

    Args:
        model (nn.Module): The PyTorch model to measure.
        input_tensor (torch.Tensor): The input tensor for inference.
        runs (int): Number of runs to average the inference time over.

    Returns:
        float: Average inference time in milliseconds.
    """

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


