"""MS²Rescore: Sensitive PSM rescoring with predicted MS² peak intensities and RTs."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from ms2rescore import MS2Rescore, __version__
from ms2rescore.config_parser import parse_configurations
from ms2rescore.exceptions import MS2RescoreConfigurationError

try:
    import matplotlib.pyplot as plt

    plt.set_loglevel("warning")
except ImportError:
    pass

LOG_MAPPING = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
LOGGER = logging.getLogger(__name__)
CONSOLE = Console(record=True)


def _build_credits():
    """Build credits."""
    text = Text()
    text.append("\n")
    text.append("MS²Rescore", style="bold link https://github.com/compomics/ms2rescore")
    text.append(f" (v{__version__})\n", style="bold")
    text.append("Developed at CompOmics, VIB / Ghent University, Belgium.\n")
    text.append("Please cite: ")
    text.append(
        "Declercq et al. MCP (2022)", style="link https://doi.org/10.1016/j.mcpro.2022.100266"
    )
    text.append("\n")
    text.stylize("cyan")
    return text


def _setup_logging(passed_level: str, log_file: Union[str, Path]):
    """Setup logging for writing to log file and Rich Console."""
    if passed_level not in LOG_MAPPING:
        raise MS2RescoreConfigurationError(
            f"Invalid log level '{passed_level}'. "
            f"Valid levels are: {', '.join(LOG_MAPPING.keys())}"
        )
    logging.basicConfig(
        format="%(name)s // %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOG_MAPPING[passed_level],
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            RichHandler(rich_tracebacks=True, console=CONSOLE, show_path=False),
        ],
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="MS²Rescore: Sensitive PSM rescoring with predicted MS²\
            peak intensities."
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument(
        "-p",
        metavar="FILE",
        action="store",
        type=str,
        dest="psm_file",
        help="path to PSM file (pin, mzid, msms.txt, tandem xml...)",
    )
    parser.add_argument(
        "-m",
        metavar="FILE",
        action="store",
        type=str,
        dest="spectrum_path",
        help="path to MGF file or directory with MGF files (default: derived from\
            identification file)",
    )
    parser.add_argument(
        "-c",
        metavar="FILE",
        action="store",
        type=str,
        dest="config_file",
        help="path to MS²Rescore configuration file (see README.md)",
    )
    parser.add_argument(
        "-t",
        metavar="PATH",
        action="store",
        type=str,
        dest="tmp_path",
        help="path to directory to place temporary files",
    )
    parser.add_argument(
        "-o",
        metavar="FILE",
        action="store",
        type=str,
        dest="output_path",
        help="name for output files (default: derive from identification file)",
    )
    parser.add_argument(
        "-l",
        metavar="LEVEL",
        action="store",
        type=str,
        dest="log_level",
        help="logging level (default: `info`)",
    )
    parser.add_argument(
        "-n",
        metavar="VALUE",
        action="store",
        type=int,
        dest="processes",
        default=None,
        help="number of parallel processes available to MS²Rescore",
    )
    parser.add_argument(
        "--psm_file_type",
        metavar="FILE",
        action="store",
        type=str,
        dest="psm_file_type",
        default=None,
        help="determines psm parser to use from PSM_utils (default: 'infer')",
    )

    return parser.parse_args()


def main():
    """Run MS²Rescore command-line interface."""
    CONSOLE.print(_build_credits())

    cli_args = _parse_arguments()
    if cli_args.config_file:
        config = parse_configurations([cli_args.config_file, cli_args])
    else:
        config = parse_configurations(cli_args)

    output_file_root = (
        Path(config["ms2rescore"]["output_path"])
        / Path(config["ms2rescore"]["psm_file"]).with_suffix("").as_uri()
    )
    _setup_logging(config["ms2rescore"]["log_level"], output_file_root + "-ms2rescore-log.txt")

    try:
        ms2rescore = MS2Rescore(configuration=config["ms2rescore"])
        ms2rescore.run()
    except Exception as e:
        LOGGER.exception(e)
        sys.exit(1)
    finally:
        CONSOLE.save_html(output_file_root + "-ms2rescore-log.html")


if __name__ == "__main__":
    main()
