"""
Profile training bottlenecks for OWM model
"""
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training_step(model, dataloader, num_iterations=10):
    """Profile training iterations"""
    model.train()
    device = next(model.parameters()).device
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("training_loop"):
            for i, batch in enumerate(dataloader):
                if i >= num_iterations:
                    break
                    
                with record_function("data_loading"):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                
                with record_function("forward"):
                    loss = model(x, y)
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer_step"):
                    # optimizer.step() would go here
                    pass
                    
    torch.cuda.synchronize()
    
    # Print results
    print("\n=== Top CPU Operations ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    print("\n=== Top CUDA Operations ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n=== By Function ===")
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=10))
    
    return prof

def benchmark_throughput(model, dataloader, num_iterations=100):
    """Measure samples/second"""
    model.train()
    device = next(model.parameters()).device
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        loss.backward()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break
        x, y = batch
        batch_size = x.shape[0]
        total_samples += batch_size
        
        x, y = x.to(device), y.to(device)
        loss = model(x, y)
        loss.backward()
        
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    throughput = total_samples / elapsed
    
    print(f"\n=== Throughput Benchmark ===")
    print(f"Total samples: {total_samples}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Time per sample: {1000*elapsed/total_samples:.2f}ms")
    
    return throughput

if __name__ == "__main__":
    print("Use this script by importing and calling:")
    print("  profile_training_step(model, dataloader)")
    print("  benchmark_throughput(model, dataloader)")
