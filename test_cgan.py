import cGAN as cg
import torch

z = cg.create_noise(8, cg.nz)
labels = torch.randint(0, 10, (8,), device=cg.device)

fake = cg.generator(z, labels)
out = cg.discriminator(fake, labels)
print(fake.shape)  # (8, 3, 32, 32)
print(out.shape)   # (8, 1, 1, 1) or (8,)