import os
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import FacadesDataset, download_facades
from models import PatchDiscriminator, UNetGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 1
LR = 2e-4
L1_LAMBDA = 100


def save_comparison(generator, source, target, path):
    generator.eval()
    with torch.no_grad():
        fake = generator(source)
    grid = torch.cat([source, fake, target], dim=0)
    save_image(grid * 0.5 + 0.5, path, nrow=source.size(0))
    generator.train()


def main():
    torch.manual_seed(42)
    random.seed(42)
    download_facades()
    shutil.rmtree("outputs", ignore_errors=True)
    os.makedirs("outputs/progress")
    os.makedirs("outputs/test_results")

    train_loader = DataLoader(
        FacadesDataset(split="train"), batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(FacadesDataset(split="val"), batch_size=5, shuffle=False)
    test_loader = DataLoader(FacadesDataset(split="test"), batch_size=5, shuffle=False)

    generator = UNetGenerator().to(DEVICE)
    discriminator = PatchDiscriminator().to(DEVICE)
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    fixed_source, fixed_target = next(iter(val_loader))
    fixed_source, fixed_target = fixed_source.to(DEVICE), fixed_target.to(DEVICE)

    print(f"device: {DEVICE}")
    print(f"generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    for epoch in range(1, EPOCHS + 1):
        g_running, d_running = 0.0, 0.0
        for source, target in train_loader:
            source, target = source.to(DEVICE), target.to(DEVICE)

            fake = generator(source)
            pred_real = discriminator(source, target)
            pred_fake = discriminator(source, fake.detach())
            loss_d = 0.5 * (
                bce(pred_real, torch.ones_like(pred_real))
                + bce(pred_fake, torch.zeros_like(pred_fake))
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            pred_fake = discriminator(source, fake)
            loss_g = bce(pred_fake, torch.ones_like(pred_fake)) + L1_LAMBDA * l1(fake, target)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_running += loss_g.item()
            d_running += loss_d.item()

        print(
            f"epoch {epoch:03d}/{EPOCHS} | "
            f"G loss {g_running / len(train_loader):.3f} | "
            f"D loss {d_running / len(train_loader):.3f}"
        )
        if epoch == 1 or epoch % 10 == 0:
            save_comparison(
                generator, fixed_source, fixed_target,
                f"outputs/progress/epoch_{epoch:03d}.png",
            )

    torch.save(generator.state_dict(), "outputs/generator.pth")

    generator.eval()
    for i, (source, target) in enumerate(test_loader):
        if i >= 4:
            break
        source, target = source.to(DEVICE), target.to(DEVICE)
        save_comparison(generator, source, target, f"outputs/test_results/batch_{i + 1}.png")

    archive_path = shutil.make_archive("pix2pix_outputs", "zip", "outputs")
    print(f"Samples archived to {archive_path}")


if __name__ == "__main__":
    main()
