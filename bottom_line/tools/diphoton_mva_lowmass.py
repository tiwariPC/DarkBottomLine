import awkward as ak
import numpy as np
import pandas as pd
import vector
import os

import onnxruntime

_default_session_options = onnxruntime.capi._pybind_state.get_default_session_options()


def get_default_session_options_new():
    _default_session_options.inter_op_num_threads = 1
    _default_session_options.intra_op_num_threads = 1
    return _default_session_options


onnxruntime.capi._pybind_state.get_default_session_options = (
    get_default_session_options_new
)


def add_diphoton_mva_inputs_for_lowmass(diphotons, events, mc_flow_corrected=False):
    diphotons["pho_lead", "ptom"] = diphotons["pho_lead"].pt / diphotons["mass"]
    diphotons["pho_sublead", "ptom"] = diphotons["pho_sublead"].pt / diphotons["mass"]

    # * sigma right vertex
    dEoE_pho1 = diphotons["pho_lead"].energyErr / diphotons["pho_lead"].energy
    dEoE_pho2 = diphotons["pho_sublead"].energyErr / diphotons["pho_sublead"].energy
    sigma_rv = 0.5 * np.sqrt(dEoE_pho1**2 + dEoE_pho2**2)
    diphotons["sigma_rv"] = sigma_rv

    # * sigma wrong vertex
    ## references:
    ## 1. https://github.com/cms-analysis/flashgg/blob/dev_legacy_runII/Taggers/plugins/DiPhotonMVAProducer.cc#L230
    ## 2. https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/bottom_line/tools/diphoton_mva.py?ref_type=heads
    def calc_displacement(photons: ak.Array, events: ak.Array) -> vector.Vector3D:
        """
        Calculate displacement for photon shower position in the calorimeter wrt PV
        """
        return vector.zip(
            {
                "x": photons.x_calo - events.PV.x,
                "y": photons.y_calo - events.PV.y,
                "z": photons.z_calo - events.PV.z,
            }
        )

    direction_pho1 = calc_displacement(diphotons.pho_lead, events)
    direction_pho2 = calc_displacement(diphotons.pho_sublead, events)

    p_pho1 = direction_pho1.unit() * diphotons.pho_lead.energyRaw
    p_pho1["energy"] = diphotons.pho_lead.energyRaw
    p4_pho1 = vector.awk(p_pho1)
    p_pho2 = direction_pho2.unit() * diphotons.pho_sublead.energyRaw
    p_pho2["energy"] = diphotons.pho_sublead.energyRaw
    p4_pho2 = vector.awk(p_pho2)

    # * cos(delta(phi))
    cos_dphi = np.cos(p4_pho1.deltaphi(p4_pho2))
    diphotons["cos_dphi"] = cos_dphi

    sech_pho1 = 1.0 / np.cosh(p4_pho1.eta)
    sech_pho2 = 1.0 / np.cosh(p4_pho2.eta)
    tanh_pho1 = np.tanh(p4_pho1.eta)
    tanh_pho2 = np.tanh(p4_pho2.eta)

    numerator_pho1 = sech_pho1 * (
        sech_pho1 * tanh_pho2 - tanh_pho1 * sech_pho2 * cos_dphi
    )
    numerator_pho2 = sech_pho2 * (
        sech_pho2 * tanh_pho1 - tanh_pho2 * sech_pho1 * cos_dphi
    )

    denominator = 1.0 - tanh_pho1 * tanh_pho2 - sech_pho1 * sech_pho2 * cos_dphi

    # beam spot sigma Z for lowmass is 3.5
    ## https://indico.cern.ch/event/1360969/contributions/5864116/attachments/2824580/4934078/2022postEE_LM_DiphotonBDT_Hgg.pdf
    beamspot_sigmaZ = 3.5
    angle_reso_wv = (-np.sqrt(2.0) * beamspot_sigmaZ / denominator) * (
        numerator_pho1 / direction_pho1.mag + numerator_pho2 / direction_pho2.mag
    )
    alpha_sig_wv = 0.5 * angle_reso_wv
    sigma_wv = np.sqrt(sigma_rv**2 + alpha_sig_wv**2)
    diphotons["sigma_wv"] = sigma_wv

    # * vertex probability
    # this in principle should have a value closer to 1 if the resolution is better,
    # by construction sigma_rv <= sigma_wv
    vtx_prob = 2 * sigma_rv / (sigma_rv + sigma_wv)
    diphotons["vtx_prob"] = vtx_prob

    # * resolution weight
    diphotons["resolution_weight"] = vtx_prob / sigma_rv + (1 - vtx_prob) / sigma_wv

    # ! if normlizing flow applied, the energyErr is corrected
    if mc_flow_corrected:
        # * sigma right vertex from NanoAOD
        dEoE_pho1_nano = (
            diphotons["pho_lead"].raw_energyErr / diphotons["pho_lead"].energy
        )
        dEoE_pho2_nano = (
            diphotons["pho_sublead"].raw_energyErr / diphotons["pho_sublead"].energy
        )
        sigma_rv_nano = 0.5 * np.sqrt(dEoE_pho1_nano**2 + dEoE_pho2_nano**2)
        diphotons["sigma_rv_nano"] = sigma_rv_nano
        # * sigma wrong vertex from NanoAOD
        sigma_wv_nano = np.sqrt(sigma_rv_nano**2 + alpha_sig_wv**2)
        diphotons["sigma_wv_nano"] = sigma_wv_nano
        # * vertex probability from NanoAOD
        vtx_prob_nano = 2 * sigma_rv_nano / (sigma_rv_nano + sigma_wv_nano)
        diphotons["vtx_prob_nano"] = vtx_prob_nano
        # * resolution weight
        diphotons["resolution_weight_nano"] = (
            vtx_prob_nano / sigma_rv_nano + (1 - vtx_prob_nano) / sigma_wv_nano
        )

    return diphotons


def get_model_path():
    model_dict = {
        "2022preEE": os.path.join(
            os.path.dirname(__file__),
            "../tools/lowmass_diphoton_mva/2022preEE/DiphotonXGboost_LM.onnx",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "../tools/lowmass_diphoton_mva/2022postEE/DiphotonXGboost_LM.onnx",
        ),
    }
    return model_dict


def get_variable_list():
    # * varible list could have:
    # * - direct variable name, e.g., sigma_wv
    # * - varible in subfield, e.g., photon_lead.eta
    variable_list = [
        "pho_lead.ptom",
        "pho_sublead.ptom",
        "pho_lead.eta",
        "pho_sublead.eta",
        "pho_lead.mvaID",
        "pho_sublead.mvaID",
        "sigma_wv",
        "cos_dphi",
    ]
    return variable_list


def eval_diphoton_mva_for_lowmass(diphotons, year="2022postEE"):
    model_dict = get_model_path()
    # model input variables
    variable_list = get_variable_list()
    dict_inputs = {
        var: ak.to_numpy(diphotons[tuple(var.split(".")) if "." in var else var])
        for var in variable_list
    }

    df_inputs = pd.DataFrame(dict_inputs)
    np_inputs = df_inputs.to_numpy().astype(np.float32)

    # fix issues mentioned here: https://github.com/microsoft/onnxruntime/issues/8313
    # ort_options = ort_session.SessionOptions()
    # ort_options = onnxruntime.capi._pybind_state.get_default_session_options()
    # ort_options.intra_op_num_threads = 1
    # ort_options.inter_op_num_threads = 1

    # create onnx session
    ort_session = onnxruntime.InferenceSession(model_dict[year])
    input_name = ort_session.get_inputs()[0].name

    # evaluation
    predictions = ort_session.run(None, {input_name: np_inputs})

    # add diphoton mva score
    # column 0: probability for class 0 -> background
    # column 1: probability for class 1 -> signal
    diphotons["diphoton_MVA"] = predictions[1][:, 1]
    diphotons["diphoton_MVA_transformed"] = (
        2.0 / (1.0 + np.exp(2.0 * np.log(1.0 / diphotons["diphoton_MVA"] - 1.0))) - 1
    )

    return diphotons
