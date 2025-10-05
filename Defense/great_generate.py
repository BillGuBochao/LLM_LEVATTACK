from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
# X, y = load_diabetes(return_X_y=True, as_frame=True)
# X["target"] = y

# syn_model = Plugins().get("great")

# syn_model.fit(X)
# syn_model.generate(count = len(X))
