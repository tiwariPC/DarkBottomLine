from bottom_line.systematics.EGM_SS_systematics import EGM_Scale_Trad, EGM_Smearing_Trad, EGM_Scale_IJazZ, EGM_Smearing_IJazZ


def Electron_Scale_Trad(pt, events, year="2022postEE", is_correction=True, restriction=None):
    """
    Applies the photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py
    """

    return EGM_Scale_Trad(pt, events, year, is_correction, restriction, is_electron=True)


def Electron_Smearing_Trad(pt, events, year="2022postEE", is_correction=True):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    return EGM_Smearing_Trad(pt, events, year, is_correction, is_electron=True)


def Electron_Scale_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G", restriction=None):
    """
    Applies the IJazZ photon pt scale corrections (use on data!) and corresponding uncertainties (on MC!).
    JSONs need to be pulled first with scripts/pull_files.py.
    The IJazZ corrections are independent and detached from the Egamma corrections.
    """

    return EGM_Scale_IJazZ(pt, events, year, is_correction, gaussians, restriction, is_electron=True)


def Electron_Smearing_IJazZ(pt, events, year="2022postEE", is_correction=True, gaussians="1G"):
    """
    Applies the photon smearing corrections and corresponding uncertainties (on MC!).
    JSON needs to be pulled first with scripts/pull_files.py
    """

    return EGM_Smearing_IJazZ(pt, events, year, is_correction, gaussians, is_electron=True)
