CONFIG_SCHEMA = {
    "gradient_clipping": (float, 1.0),
    "weight_decay": (float, 0.01),
    "early_stopping": (bool, True),
    "early_stopping_patience": (int, 8),
    "save_every": (int, 0),
    "num_workers": (int, 4),
    "pin_memory": (bool, True),
    "use_mixed_precision": (bool, False),
    "repetition_penalty": (float, 1.1),
    "max_response_length": (int, 64),
}
