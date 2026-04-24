# benchmark_torch.py
import torch, time
from engine.core import YAMLConfig

CONFIG = "configs/deimv2/deimv2_hgnetv2_n_fisheye.yml"
WEIGHTS = "/home/vdlung/nas/vdlung/LocTH/omni2rect_DEIM/DEIMv2/outputs/deimv2_hgnetv2_n_fisheye/best_stg1.pth"
INPUT_SIZE = (1, 3, 640, 640)
WARMUP = 50
RUNS = 300
DEVICE = torch.device("cuda:0")

cfg = YAMLConfig(CONFIG, resume=WEIGHTS)
model = cfg.model.deploy()      # sets model to eval + deploy mode
model = model.to(DEVICE).eval()

dummy = torch.randn(*INPUT_SIZE).to(DEVICE)

# Warmup
with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(dummy)

torch.cuda.synchronize()

# Benchmark
times = []
with torch.no_grad():
    for _ in range(RUNS):
        start = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

avg_ms = (sum(times) / len(times)) * 1000
fps    = 1000 / avg_ms

print(f"PyTorch Inference")
print(f"  Avg latency : {avg_ms:.2f} ms")
print(f"  FPS         : {fps:.1f}")