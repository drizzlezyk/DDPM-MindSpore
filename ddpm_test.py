import ddpm


model = ddpm.Unet(
    dim=16,
    out_dim=3,
    dim_mults=(1, 2, 4, 8)
)

diffusion = ddpm.GaussianDiffusion(
    model,
    image_size=16,
    timesteps=20,             # number of steps
    sampling_timesteps=10,     # number of sampling time steps
    loss_type='l1'            # L1 or L2
)

trainer = ddpm.Trainer(
    diffusion,
    "C:\\Users\\Administrator\\PycharmProjects\\DDPM\\datasets\\test",
    train_batch_size=1,
    train_lr=8e-5,
    train_num_steps=101,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    save_and_sample_every=1,
    num_samples=4
)

trainer.train()
