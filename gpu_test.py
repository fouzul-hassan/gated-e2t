import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# import torch
# print(torch.__version__)  # Check PyTorch version
# print(torch.cuda.is_available())  # Check if CUDA is available