import torch 
import torch.nn as nn
import torchvision.transforms as T
import PIL
from torch.utils.data import DataLoader
from load_data import CelebADataset
from model import Generator, Discriminator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
REAL = 1
FAKE = 0 


if __name__ == '__main__':

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    batch_size = 128

    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.002, betas=[0.5, 0.999])
    disc_opt = torch.optim.Adam(disc.parameters(), lr=0.002, betas=[0.5, 0.999])

    dataloader = DataLoader(CelebADataset(img_size=(64,64)), batch_size = batch_size, shuffle=True, num_workers=24)

    loss_fn = nn.BCELoss()
    counter = 0 

    test_vector = torch.rand((1, 100, 1, 1), device=device)
    for epoch in range(100):
        for image, label in dataloader:
            image = image.to(device)
            if image.size(0) != batch_size:
                break
            # train discriminator

            disc.zero_grad()
            gen_input = torch.randn((batch_size, 100, 1, 1), device=device)
            fake_im = gen(gen_input)

            disc_out = torch.cat((disc(image).view(-1), disc(fake_im).view(-1)), dim=-1)
            disc_label = torch.cat((torch.full((batch_size,), REAL, dtype=torch.float32, device=device), torch.full((batch_size,), FAKE, dtype=torch.float32, device=device)))
            disc_loss = loss_fn(disc_out, disc_label)
            disc_loss.backward()
            disc_opt.step()
            disc_opt.zero_grad()
            
            # train generator
            gen.zero_grad()

            gen_out = gen(torch.randn(batch_size*2, 100, 1, 1, device=device))
            disc_out = disc(gen_out).view(-1)
            gen_label = torch.full((batch_size*2, ), REAL, dtype=torch.float32, device=device)
            gen_loss = loss_fn(disc_out, gen_label)
            gen_loss.backward()
            gen_opt.step()
            gen_opt.zero_grad()
            gen.zero_grad()
            print(counter, gen_loss.mean().item(), disc_loss.mean().item())
            counter += 1
            if counter % 100 == 0:
                with torch.no_grad():
                    test_im = gen(test_vector)[0]
                    test_im = (test_im + 1)/2
                    print(test_im)
                    im: PIL.Image = T.ToPILImage()(test_im)
                    im.save(f'test_ims/{counter}.png')