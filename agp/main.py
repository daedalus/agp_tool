from .api import generate_report
from .cli import parse_args


def main():
    args = parse_args()
    generate_report(
        input_file=args.input_file,
        output=args.output,
        very_low_threshold=args.very_low_threshold,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        very_high_threshold=args.very_high_threshold,
        tight_low=args.tight_low,
        tight_high=args.tight_high,
        bin_minutes=args.bin_minutes,
        sensor_interval=args.sensor_interval,
        min_samples=args.min_samples,
        no_plot=args.no_plot,
        verbose=args.verbose,
        export=args.export,
        config=args.config,
        patient_name=args.patient_name,
        patient_id=args.patient_id,
        doctor=args.doctor,
        notes=args.notes,
        heatmap=args.heatmap,
        heatmap_cmap=args.heatmap_cmap,
        pdf=args.pdf,
        show=True,
        close=True,
    )


if __name__ == "__main__":
    main()

