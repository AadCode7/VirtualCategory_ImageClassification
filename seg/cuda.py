import torch

def check_torch_version():
    print("PyTorch Version: ", torch.__version__)
    print("CUDA Version: ", torch.version.cuda)
    print("cuDNN Version: ", torch.backends.cudnn.version())
    print("Is CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs: ", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} Name: ", torch.cuda.get_device_name(i))
    else:
        print("No CUDA-capable GPU detected")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    check_torch_version()
