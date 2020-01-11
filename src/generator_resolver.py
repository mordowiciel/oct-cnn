def resolve_generators(cfg, training_image_datagen, test_image_datagen):
    generator_seed = 42
    if cfg.network.architecture == 'VGG-16-TL':
        training_generator = training_image_datagen.flow_from_directory(
            directory=cfg.dataset.training_dataset_path,
            target_size=cfg.dataset.img_size,
            batch_size=cfg.training.training_batch_size,
            interpolation='bilinear',
            color_mode='rgb',
            seed=generator_seed,
            shuffle=True
        )
        test_generator = test_image_datagen.flow_from_directory(
            directory=cfg.dataset.test_dataset_path,
            target_size=cfg.dataset.img_size,
            batch_size=cfg.training.test_batch_size,
            interpolation='bilinear',
            color_mode='rgb',
            seed=generator_seed,
            shuffle=False
        )
    else:
        training_generator = training_image_datagen.flow_from_directory(
            directory=cfg.dataset.training_dataset_path,
            target_size=cfg.dataset.img_size,
            batch_size=cfg.training.training_batch_size,
            interpolation='bilinear',
            color_mode='grayscale',
            seed=generator_seed,
            shuffle=True
        )
        test_generator = test_image_datagen.flow_from_directory(
            directory=cfg.dataset.test_dataset_path,
            target_size=cfg.dataset.img_size,
            batch_size=cfg.training.test_batch_size,
            interpolation='bilinear',
            color_mode='grayscale',
            seed=generator_seed,
            shuffle=False
        )

    return training_generator, test_generator
