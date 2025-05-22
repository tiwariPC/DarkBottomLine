import numpy as np
import awkward as ak


def add_photon_SC_eta(photons: ak.Array, PV: ak.Array) -> ak.Array:
    """
    Add supercluster eta to photon object, following the implementation from https://github.com/bartokm/GbbMET/blob/026dac6fde5a1d449b2cfcaef037f704e34d2678/analyzer/Analyzer.h#L2487
    In the current NanoAODv11, there is only the photon eta which is the SC eta corrected by the PV position.
    The SC eta is needed to correctly apply a number of corrections and systematics.
    """

    if "superclusterEta" in photons.fields:
        photons["ScEta"] = photons.superclusterEta
        return photons

    PV_x = PV.x.to_numpy()
    PV_y = PV.y.to_numpy()
    PV_z = PV.z.to_numpy()

    mask_barrel = photons.isScEtaEB
    mask_endcap = photons.isScEtaEE

    tg_theta_over_2 = np.exp(-photons.eta)
    # avoid dividion by zero
    tg_theta_over_2 = np.where(tg_theta_over_2 == 1., 1 - 1e-10, tg_theta_over_2)
    tg_theta = 2 * tg_theta_over_2 / (1 - tg_theta_over_2 * tg_theta_over_2)  # tg(a+b) = tg(a)+tg(b) / (1-tg(a)*tg(b))

    # calculations for EB
    R = 130.
    angle_x0_y0 = np.zeros_like(PV_x)

    angle_x0_y0[PV_x > 0] = np.arctan(PV_y[PV_x > 0] / PV_x[PV_x > 0])
    angle_x0_y0[PV_x < 0] = np.pi + np.arctan(PV_y[PV_x < 0] / PV_x[PV_x < 0])
    angle_x0_y0[((PV_x == 0) & (PV_y >= 0))] = np.pi / 2
    angle_x0_y0[((PV_x == 0) & (PV_y < 0))] = -np.pi / 2

    alpha = angle_x0_y0 + (np.pi - photons.phi)
    sin_beta = np.sqrt(PV_x**2 + PV_y**2) / R * np.sin(alpha)
    beta = np.abs(np.arcsin(sin_beta))
    gamma = np.pi / 2 - alpha - beta
    length = np.sqrt(R**2 + PV_x**2 + PV_y**2 - 2 * R * np.sqrt(PV_x**2 + PV_y**2) * np.cos(gamma))
    z0_zSC = length / tg_theta

    tg_sctheta = np.copy(tg_theta)
    # correct values for EB
    tg_sctheta = ak.where(mask_barrel, R / (PV_z + z0_zSC), tg_sctheta)

    # calculations for EE
    intersection_z = np.where(photons.eta > 0, 310., -310.)
    base = intersection_z - PV_z
    r = base * tg_theta
    crystalX = PV_x + r * np.cos(photons.phi)
    crystalY = PV_y + r * np.sin(photons.phi)
    # correct values for EE
    tg_sctheta = ak.where(
        mask_endcap, np.sqrt(crystalX**2 + crystalY**2) / intersection_z, tg_sctheta
    )

    sctheta = np.arctan(tg_sctheta)
    sctheta = ak.where(
        sctheta < 0, np.pi + sctheta, sctheta
    )
    ScEta = -np.log(
        np.tan(sctheta / 2)
    )

    photons["ScEta"] = ScEta

    return photons
