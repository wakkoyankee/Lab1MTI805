import torch
from dataset import CastleEldenDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from torch.utils.tensorboard import SummaryWriter

def train_fn(disc_C, disc_E, gen_E, gen_C, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)
    global writer
    loss = []

    for idx, (elden, castle) in enumerate(loop):
        elden = elden.to(config.DEVICE)
        castle = castle.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_castle = gen_C(elden)
            D_C_real = disc_C(castle)
            D_C_fake = disc_C(fake_castle.detach())
            C_reals += D_C_real.mean().item()
            C_fakes += D_C_fake.mean().item()
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss

            fake_elden = gen_E(castle)
            D_E_real = disc_E(elden)
            D_E_fake = disc_E(fake_elden.detach())
            D_E_real_loss = mse(D_E_real, torch.ones_like(D_E_real))
            D_E_fake_loss = mse(D_E_fake, torch.zeros_like(D_E_fake))
            D_E_loss = D_E_real_loss + D_E_fake_loss

            # put it togethor
            D_loss = (D_C_loss + D_E_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_C_fake = disc_C(fake_castle)
            D_E_fake = disc_E(fake_elden)
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))
            loss_G_E = mse(D_E_fake, torch.ones_like(D_E_fake))

            # cycle loss
            cycle_elden = gen_E(fake_castle)
            cycle_castle = gen_C(fake_elden)
            cycle_elden_loss = l1(elden, cycle_elden)
            cycle_castle_loss = l1(castle, cycle_castle)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_elden = gen_E(elden)
            identity_castle = gen_C(castle)
            identity_elden_loss = l1(elden, identity_elden)
            identity_castle_loss = l1(castle, identity_castle)

            # add all together
            G_loss = (
                loss_G_E
                + loss_G_C
                + cycle_elden_loss * config.LAMBDA_CYCLE
                + cycle_castle_loss * config.LAMBDA_CYCLE
                + identity_castle_loss * config.LAMBDA_IDENTITY
                + identity_elden_loss * config.LAMBDA_IDENTITY
            )

            loss.append(G_loss)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_castle*0.5+0.5, f"saved_images/castle_{idx}.png")
            save_image(fake_elden*0.5+0.5, f"saved_images/elden_{idx}.png")

        loop.set_postfix(C_real=C_reals/(idx+1), C_fake=C_fakes/(idx+1))
    
    writer.add_scalar("Loss/train", np.array(loss).mean(), epoch)



def main():
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    disc_E = Discriminator(in_channels=3).to(config.DEVICE)
    gen_E = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_C.parameters()) + list(disc_E.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_E.parameters()) + list(gen_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_C, gen_C, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_E, gen_E, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_C, disc_C, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_E, disc_E, opt_disc, config.LEARNING_RATE,
        )

    dataset = CastleEldenDataset(
        root_castle=config.TRAIN_DIR+"/castle", root_elden=config.TRAIN_DIR+"/elden", transform=config.transforms
    )
    val_dataset = CastleEldenDataset(
       root_castle=config.VAL_DIR + "/castle", root_elden=config.VAL_DIR +"/elden", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_C, disc_E, gen_E, gen_C, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_C)
            save_checkpoint(gen_E, opt_gen, filename=config.CHECKPOINT_GEN_E)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_C)
            save_checkpoint(disc_E, opt_disc, filename=config.CHECKPOINT_CRITIC_E)
    
if __name__ == "__main__":
    global writer
    writer = SummaryWriter()
    main()
    writer.flush()
    