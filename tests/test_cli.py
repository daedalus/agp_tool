import argparse
import pytest

from agp.cli import build_parser


def test_build_parser_returns_argument_parser():
    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_default_low_threshold():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.low_threshold == 70


def test_default_high_threshold():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.high_threshold == 180


def test_default_very_low_threshold():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.very_low_threshold == 54


def test_default_very_high_threshold():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.very_high_threshold == 250


def test_default_bin_minutes():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.bin_minutes == 5


def test_default_sensor_interval():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.sensor_interval == 5


def test_no_plot_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.no_plot is False


def test_no_plot_set_when_passed():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx", "--no-plot"])
    assert args.no_plot is True


def test_verbose_defaults_false():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx"])
    assert args.verbose is False


def test_verbose_set_when_passed():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx", "--verbose"])
    assert args.verbose is True


def test_custom_threshold_overrides_default():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx", "--high-threshold", "200"])
    assert args.high_threshold == 200


def test_custom_bin_minutes():
    parser = build_parser()
    args = parser.parse_args(["dummy.xlsx", "--bin-minutes", "15"])
    assert args.bin_minutes == 15
