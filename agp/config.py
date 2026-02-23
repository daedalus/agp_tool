def build_config(args):
    """Return a dict of all threshold/config constants derived from parsed args."""
    return {
        "VERY_LOW": args.very_low_threshold,
        "LOW": args.low_threshold,
        "HIGH": args.high_threshold,
        "VERY_HIGH": args.very_high_threshold,
        "TIGHT_LOW": args.tight_low,
        "TIGHT_HIGH": args.tight_high,
        "BIN_MINUTES": args.bin_minutes if args.bin_minutes > 0 else 1,
        "MIN_SAMPLES_PER_BIN": args.min_samples,
        "SENSOR_INTERVAL": args.sensor_interval if args.sensor_interval > 0 else 5,
        "ROC_CLIP": 10,  # Physiological constant
    }
