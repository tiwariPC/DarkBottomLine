import numpy as np


def sigma_m(leading_photon_energyErr, subleading_photon_energyErr, leading_photon_energy, subleading_photon_energy):

    return 0.5 * np.sqrt((leading_photon_energyErr / leading_photon_energy) ** 2 + (subleading_photon_energyErr / subleading_photon_energy) ** 2)


def sigma_m_smeared(leading_photon_energyErr, subleading_photon_energyErr, leading_photon_energy, subleading_photon_energy, leading_photon_smearing_term, subleading_photon_smearing_term):
    """
    This function computes the sigma_m variable for the diphoton system.
    Used when we have smeared sigma_m values.
    """

    return 0.5 * np.sqrt((np.sqrt((leading_photon_energyErr)**2 + (leading_photon_smearing_term * ((leading_photon_energy))) ** 2) / (leading_photon_energy)) ** 2 + (np.sqrt((subleading_photon_energyErr) ** 2 + (subleading_photon_smearing_term * ((subleading_photon_energy))) ** 2) / (subleading_photon_energy)) ** 2)


def compute_sigma_m(diphotons, processor='base', flow_corrections=False, smear=True, IsData=False):
    """
    This function computes the sigma_m variable for the diphoton system.

    Since we have the base and tag and probe processors, we have to account for the different variables names.
    Such as "pho_lead" and "tag" and "probe" for the base and tag and probe processors, respectively.

    Thats is why we have the processor argument.
    """
    available_processors = ['base','tnp']

    if processor not in available_processors:
        raise ValueError("Specify a valid processor: 'base' or 'tnp'")

    if processor == 'base':

        # Here we have the possibility of four sigma_m variables
        # - the nominal
        # - the smeared
        # - the "nominal" corrected
        # - the corrected + smearing term

        # Lets start by the nominal!
        if flow_corrections and not IsData:
            diphotons["sigma_m_over_m"] = sigma_m(diphotons.pho_lead.raw_energyErr, diphotons.pho_sublead.raw_energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta))
            diphotons["sigma_m_over_m_corr"] = sigma_m(diphotons.pho_lead.energyErr, diphotons.pho_sublead.energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta))
            if smear:
                diphotons["sigma_m_over_m_Smeared_corr"] = sigma_m_smeared(diphotons.pho_lead.energyErr, diphotons.pho_sublead.energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta), diphotons["pho_lead"].rho_smear, diphotons["pho_sublead"].rho_smear)
                diphotons["sigma_m_over_m_Smeared"] = sigma_m_smeared(diphotons.pho_lead.raw_energyErr, diphotons.pho_sublead.raw_energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta), diphotons["pho_lead"].rho_smear, diphotons["pho_sublead"].rho_smear)
        else:
            diphotons["sigma_m_over_m"] = sigma_m(diphotons.pho_lead.energyErr, diphotons.pho_sublead.energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta))
            if smear:
                diphotons["sigma_m_over_m_Smeared"] = sigma_m_smeared(diphotons.pho_lead.energyErr, diphotons.pho_sublead.energyErr, diphotons["pho_lead"].pt * np.cosh(diphotons["pho_lead"].eta), diphotons["pho_sublead"].pt * np.cosh(diphotons["pho_sublead"].eta), diphotons["pho_lead"].rho_smear, diphotons["pho_sublead"].rho_smear)

    elif processor == 'tnp':
        # Lets start by the nominal!
        if flow_corrections and not IsData:
            diphotons["sigma_m_over_m"] = sigma_m(diphotons.tag.raw_energyErr, diphotons.probe.raw_energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta))
            diphotons["sigma_m_over_m_corr"] = sigma_m(diphotons.tag.energyErr, diphotons.probe.energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta))
            if smear:
                diphotons["sigma_m_over_m_Smeared"] = sigma_m_smeared(diphotons.tag.raw_energyErr, diphotons.probe.raw_energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta), diphotons["tag"].rho_smear, diphotons["probe"].rho_smear)
                diphotons["sigma_m_over_m_Smeared_corr"] = sigma_m_smeared(diphotons.tag.energyErr, diphotons.probe.energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta), diphotons["tag"].rho_smear, diphotons["probe"].rho_smear)
        else:
            diphotons["sigma_m_over_m"] = sigma_m(diphotons.tag.energyErr, diphotons.probe.energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta))
            if smear:
                diphotons["sigma_m_over_m_Smeared"] = sigma_m_smeared(diphotons.tag.energyErr, diphotons.probe.energyErr, diphotons["tag"].pt * np.cosh(diphotons["tag"].eta), diphotons["probe"].pt * np.cosh(diphotons["probe"].eta), diphotons["tag"].rho_smear, diphotons["probe"].rho_smear)
    else:
        print("Specify a valid processor: base,tnp")
        exit()

    return diphotons
