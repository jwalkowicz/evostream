import locale

import matplotlib as mpl


def use_polish_number_format() -> None:
    locale.setlocale(locale.LC_NUMERIC, "pl_PL.UTF-8")
    mpl.rcParams["axes.formatter.use_locale"] = True
