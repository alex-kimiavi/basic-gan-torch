import torch 
import torchvision.transforms as T
from model import Generator
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(f'models/2600.pth'))
    num_gen = 10
    for i in range(10):
        vec = torch.randn(1, 100, 1, 1, device=device)
        out = gen(vec)[0]
        out_im = T.ToPILImage()(out)
        out_im.save(f'{i}.png')