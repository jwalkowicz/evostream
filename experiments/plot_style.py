import locale

import matplotlib as mpl


def use_polish_number_format() -> None:
    """Axis tick labels with a decimal comma and a space as the thousands
    separator (0,25; 10 000), as in the Polish thesis text."""
    locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")
    mpl.rcParams["axes.formatter.use_locale"] = True
