from typing import List, Optional, Tuple

import awkward as ak
import numpy
import xgboost


def calculate_diphoton_mva(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    diphotons: ak.Array,
    events: ak.Array,
) -> ak.Array:
    """
    Calculate DiphotonID bdt scores for diphoton.
    """
    if mva[0] is None:
        return diphotons
    diphoton_mva = mva[0]

    var_order = mva[1]

    bdt_vars = {}

    bdt_vars["dipho_leadIDMVA"] = diphotons.pho_lead.mvaID
    bdt_vars["dipho_subleadIDMVA"] = diphotons.pho_sublead.mvaID
    bdt_vars["dipho_leadEta"] = diphotons.pho_lead.eta
    bdt_vars["dipho_subleadEta"] = diphotons.pho_sublead.eta
    bdt_vars["dipho_lead_ptoM"] = diphotons.pho_lead.pt / diphotons.mass
    bdt_vars["dipho_sublead_ptoM"] = diphotons.pho_sublead.pt / diphotons.mass

    def calc_displacement(
        photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        x = photons.x_calo - events.PV.x
        y = photons.y_calo - events.PV.y
        z = photons.z_calo - events.PV.z
        return ak.zip({"x": x, "y": y, "z": z}, with_name="Vector3D")

    v_lead = calc_displacement(diphotons.pho_lead, events)
    v_sublead = calc_displacement(diphotons.pho_sublead, events)

    p_lead = v_lead.unit() * diphotons.pho_lead.energyRaw
    p_lead["energy"] = diphotons.pho_lead.energyRaw
    p_lead = ak.with_name(p_lead, "Momentum4D")
    p_sublead = v_sublead.unit() * diphotons.pho_sublead.energyRaw
    p_sublead["energy"] = diphotons.pho_sublead.energyRaw
    p_sublead = ak.with_name(p_sublead, "Momentum4D")

    sech_lead = 1.0 / numpy.cosh(p_lead.eta)
    sech_sublead = 1.0 / numpy.cosh(p_sublead.eta)
    tanh_lead = numpy.cos(p_lead.theta)
    tanh_sublead = numpy.cos(p_sublead.theta)

    cos_dphi = numpy.cos(p_lead.deltaphi(p_sublead))

    numerator_lead = sech_lead * (
        sech_lead * tanh_sublead - tanh_lead * sech_sublead * cos_dphi
    )
    numerator_sublead = sech_sublead * (
        sech_sublead * tanh_lead - tanh_sublead * sech_lead * cos_dphi
    )

    denominator = 1.0 - tanh_lead * tanh_sublead - sech_lead * sech_sublead * cos_dphi

    add_reso = (
        0.5
        * (-numpy.sqrt(2.0) * events.BeamSpot.sigmaZ / denominator)
        * (numerator_lead / v_lead.mag + numerator_sublead / v_sublead.mag)
    )

    dEnorm_lead = diphotons.pho_lead.energyErr / diphotons.pho_lead.energy
    dEnorm_sublead = diphotons.pho_sublead.energyErr / diphotons.pho_sublead.energy

    sigma_m = 0.5 * numpy.sqrt(dEnorm_lead ** 2 + dEnorm_sublead ** 2)
    sigma_wv = numpy.sqrt(add_reso ** 2 + sigma_m ** 2)

    vtx_prob = ak.full_like(sigma_m, 0.999)  # !!!! placeholder !!!!

    bdt_vars["CosPhi"] = cos_dphi
    bdt_vars["vtxprob"] = vtx_prob
    bdt_vars["sigmarv"] = sigma_m
    bdt_vars["sigmawv"] = sigma_wv

    counts = ak.num(diphotons, axis=-1)
    bdt_inputs = numpy.column_stack(
        [ak.to_numpy(ak.flatten(bdt_vars[name])) for name in var_order]
    )
    tempmatrix = xgboost.DMatrix(bdt_inputs, feature_names=var_order)
    scores = diphoton_mva.predict(tempmatrix)

    for var, arr in bdt_vars.items():
        if "dipho" not in var:
            diphotons[var] = arr

    diphotons["bdt_score"] = ak.unflatten(scores, counts)

    return diphotons


def calculate_retrained_diphoton_mva(
    self,
    mva: Tuple[Tuple[Optional[xgboost.Booster], Optional[xgboost.Booster]], List[str]],
    diphotons: ak.Array,
    events: ak.Array,
) -> ak.Array:

    if self.analysis != "tagAndProbe":
        pho_lead = "pho_lead"
        pho_sublead = "pho_sublead"
    else:
        pho_lead = "tag"
        pho_sublead = "probe"

    """
    Calculate DiphotonID bdt scores for diphoton using retrained bdt that avoids flashgg inputs.
    To calculate the score I need some extra variables.
    """
    if len(events) == 0 or mva[0] is None:
        diphotons["bdt_score"] = ak.zeros_like(diphotons.mass)
        return diphotons, events

    diphoton_mva = []
    diphoton_mva.append(mva[0][0])
    diphoton_mva.append(mva[0][1])

    var_order = mva[1]

    events_bdt = events

    # changed naming to match one used in retraining of the bdt
    events_bdt["LeadPhoton_mvaID"] = diphotons[pho_lead].mvaID
    events_bdt["SubleadPhoton_mvaID"] = diphotons[pho_sublead].mvaID
    events_bdt["LeadPhoton_eta"] = diphotons[pho_lead].eta
    events_bdt["SubleadPhoton_eta"] = diphotons[pho_sublead].eta
    events_bdt["LeadPhoton_pt_mgg"] = diphotons[pho_lead].pt / diphotons.mass
    events_bdt["SubleadPhoton_pt_mgg"] = diphotons[pho_sublead].pt / diphotons.mass

    def calc_displacement(
        photons: ak.Array, events: ak.Array
    ) -> ak.Array:
        """
        Calculate displacement for photon shower position in the calorimeter wrt PV
        """
        x = photons.x_calo - events.PV.x
        y = photons.y_calo - events.PV.y
        z = photons.z_calo - events.PV.z
        return ak.zip({"x": x, "y": y, "z": z}, with_name="Vector3D")

    v_lead = calc_displacement(diphotons[pho_lead], events)
    v_sublead = calc_displacement(diphotons[pho_sublead], events)

    p_lead = v_lead.unit() * diphotons[pho_lead].energy
    p_lead["energy"] = diphotons[pho_lead].energy
    p_lead = ak.with_name(p_lead, "Momentum4D")
    p_sublead = v_sublead.unit() * diphotons[pho_sublead].energy
    p_sublead["energy"] = diphotons[pho_sublead].energy
    p_sublead = ak.with_name(p_sublead, "Momentum4D")

    sech_lead = 1.0 / numpy.cosh(p_lead.eta)
    sech_sublead = 1.0 / numpy.cosh(p_sublead.eta)
    tanh_lead = numpy.cos(p_lead.theta)
    tanh_sublead = numpy.cos(p_sublead.theta)

    cos_dphi = numpy.cos(p_lead.deltaphi(p_sublead))

    numerator_lead = sech_lead * (
        sech_lead * tanh_sublead - tanh_lead * sech_sublead * cos_dphi
    )
    numerator_sublead = sech_sublead * (
        sech_sublead * tanh_lead - tanh_sublead * sech_lead * cos_dphi
    )

    denominator = 1.0 - tanh_lead * tanh_sublead - sech_lead * sech_sublead * cos_dphi

    add_reso = (
        0.5
        * (-numpy.sqrt(2.0) * events.BeamSpot.sigmaZ / denominator)
        * (numerator_lead / v_lead.mag + numerator_sublead / v_sublead.mag)
    )

    dEnorm_lead = diphotons[pho_lead].energyErr / diphotons[pho_lead].energy
    dEnorm_sublead = diphotons[pho_sublead].energyErr / diphotons[pho_sublead].energy

    sigma_m = 0.5 * numpy.sqrt(dEnorm_lead**2 + dEnorm_sublead**2)
    sigma_wv = numpy.sqrt(add_reso**2 + sigma_m**2)

    # !!!! placeholder !!!!
    # this in principle should have a value closer to 1 if the resolution is better,
    # by construction sigma_m <= sigma_wv
    vtx_prob = 2 * sigma_m / (sigma_m + sigma_wv)

    # z coordinate of primary vertices other than the main one
    # padded to have at least 3 entry for each event (useful for slicing)
    OtherPV_z = ak.to_numpy(
        ak.fill_none(ak.pad_none(events.OtherPV.z, 3, axis=1), -999.0)
    )
    PV_z = ak.to_numpy(events.PV.z)
    events.OtherPV.z = ak.from_numpy(OtherPV_z)
    # reshaping to match OtherPV_z
    PV_z = numpy.full_like(
        numpy.arange(3 * len(PV_z)).reshape(len(PV_z), 3), 1, dtype=float
    )
    PV_z[:, 0] = PV_z[:, 0] * ak.fill_none(events.PV.z, -9999.0)
    PV_z[:, 1] = PV_z[:, 1] * ak.fill_none(events.PV.z, -9999.0)
    PV_z[:, 2] = PV_z[:, 2] * ak.fill_none(events.PV.z, -9999.0)

    # z distance of the first three PVs from the main one
    events["OtherPV_dZ_0"] = ak.from_numpy(numpy.abs(PV_z - OtherPV_z))

    events_bdt["Diphoton_cos_dPhi"] = cos_dphi
    events_bdt["PV_score"] = events.PV.score
    events_bdt["PV_chi2"] = events.PV.chi2
    events_bdt["nPV"] = events.PV.npvs
    for i in range(len(OtherPV_z[0])):
        name = "%s_%d" % ("dZ", i + 1)
        events_bdt[name] = events.OtherPV_dZ_0[:, i]

    events_bdt["vtxProb"] = vtx_prob
    events_bdt["sigmaMrv"] = sigma_m
    events_bdt["sigmaMwv"] = sigma_wv

    for name in var_order:
        events_bdt[name] = ak.fill_none(events_bdt[name], -999.0)

    bdt_features = []
    for x in var_order:
        if isinstance(x, tuple):
            name_flat = "_".join(x)
            events_bdt[name_flat] = events_bdt[x]
            bdt_features.append(name_flat)
        else:
            bdt_features.append(x)

    events_bdt = ak.values_astype(events_bdt, numpy.float64)
    features_bdt = ak.to_numpy(events_bdt[bdt_features])

    features_bdt_matrix = xgboost.DMatrix(
        features_bdt.view((float, len(features_bdt.dtype.names)))
    )

    scores = []
    for bdt in diphoton_mva:
        scores.append(bdt.predict(features_bdt_matrix))

    for var in bdt_features:
        if "dipho" not in var:
            diphotons[var] = events_bdt[var]

    scores_out = ak.where(
        events.event % 2 < 1,
        scores[1],
        scores[0]
    )

    diphotons["bdt_score"] = ak.zeros_like(diphotons.mass)
    diphotons["bdt_score"] = scores_out

    return diphotons, events
