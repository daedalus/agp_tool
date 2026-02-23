from .cli import parse_args
from .config import build_config
from .data import load_and_preprocess
from .metrics import compute_all_metrics
from .plot import build_agp_profile, generate_agp_plot
from .report import create_report_header, print_clinical_summary
from .export import export_metrics
from .pdf import png_to_pdf


def main():
    args = parse_args()
    cfg = build_config(args)
    report_header = create_report_header(args)
    df = load_and_preprocess(args.input_file, cfg, verbose=args.verbose)
    metrics = compute_all_metrics(df, cfg)
    if not args.no_plot:
        result = build_agp_profile(df, cfg)
        generate_agp_plot(df, result, metrics, cfg, args, report_header)
        if getattr(args, "pdf", False):
            pdf_path = args.output.rsplit(".", 1)[0] + ".pdf"
            png_to_pdf(args.output, pdf_path)
            if args.verbose:
                print(f"PDF saved to: {pdf_path}")
    elif args.verbose:
        print("Plot generation skipped (--no-plot flag used)")
    print_clinical_summary(metrics, report_header, cfg)
    if args.export:
        export_metrics(
            metrics, args.export, report_header=report_header, verbose=args.verbose
        )


if __name__ == "__main__":
    main()
