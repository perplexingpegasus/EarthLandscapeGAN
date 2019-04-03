from data.model_trainer import ScheduledParam


progan_config = {
    'channels': [512, 512, 512, 512, 256, 128, 64, 32, 16],
    'batch_size': ScheduledParam(
        {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 8, 512: 4, 1024: 3}),
    'batch_repeats': 4,
    'imgs_per_stage': 800000,
    'g_learning_rate': 0.001,
    'g_beta1': 0.0,
    'g_beta2': 0.99,
    'd_learning_rate': 0.001,
    'd_beta1': 0.0,
    'd_beta2': 0.99,
    'gp_lambda': 10.0,
    'gp_gamma': 1.0,
    'z_length': 512,
    'ema_decay': 0.999,
}

stylegan_config = progan_config.copy()
stylegan_config.update({
    'batch_repeats': ScheduledParam(
        {4: 4, 8: 4, 16: 4, 32: 4, 64: 4, 128: 4, 256: 4, 512: 4, 1024: 4}),
    'g_learning_rate': ScheduledParam(
        {4: 0.001, 8: 0.001, 16: 0.001, 32: 0.001, 64: 0.001, 128: 0.001, 256: 0.0015, 512: 0.002, 1024: 0.003}),
    'd_learning_rate': ScheduledParam(
        {4: 0.001, 8: 0.001, 16: 0.001, 32: 0.001, 64: 0.001, 128: 0.001, 256: 0.0015, 512: 0.002, 1024: 0.003}),
    'mapping_layers': 8,
    'mapping_size': 512,
    'gp_lambda': 5.0,
    'mapping_scale': 0.01,
    'z_fixed_size': 6
})