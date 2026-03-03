from .api import generate_report
from .cli import parse_args


def main():
    args = parse_args()
    kwargs = vars(args)
    kwargs.pop("version", None)  # --version is not a parameter of generate_report
    generate_report(**kwargs, show=True, close=True)


if __name__ == "__main__":
    main()
