import torch

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print('done')