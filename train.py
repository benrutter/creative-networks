from gan import ArtDCGAN

heist_bot = ArtDCGAN(size=1)
heist_bot.build_gan()
heist_bot.train(5000, image_save_interval=50, model_save_interval=100)
