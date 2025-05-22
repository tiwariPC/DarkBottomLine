import awkward as ak
import numpy as np
import pandas as pd
import os

import onnxruntime

# ref: https://github.com/microsoft/onnxruntime/issues/8313
_default_session_options = onnxruntime.capi._pybind_state.get_default_session_options()


def get_default_session_options_new():
    _default_session_options.inter_op_num_threads = 1
    _default_session_options.intra_op_num_threads = 1
    return _default_session_options


onnxruntime.capi._pybind_state.get_default_session_options = (
    get_default_session_options_new
)


def get_model_path():
    model_dict = {
        "2022preEE": os.path.join(
            os.path.dirname(__file__),
            "../tools/lowmass_dykiller/2022preEE/NN.onnx",
        ),
        "2022postEE": os.path.join(
            os.path.dirname(__file__),
            "../tools/lowmass_dykiller/2022postEE/NN.onnx",
        ),
    }
    return model_dict


def get_variable_list():
    # * varible list could have:
    # * - direct variable name, e.g., sigma_wv
    # * - varible in subfield, e.g., photon_lead.eta
    variable_list = [
        "ptom",
        "pho_lead.s4",
        "pho_lead.sieip",
        "pho_lead.sipip",
        "pho_lead.sieie",
        "pho_lead.phi",
        "pho_lead.phiWidth",
        "pho_lead.ptom",
        "pho_lead.r9",
        "pho_lead.mvaID",
        "pho_sublead.s4",
        "pho_sublead.sieip",
        "pho_sublead.sipip",
        "pho_sublead.sieie",
        "pho_sublead.phi",
        "pho_sublead.phiWidth",
        "pho_sublead.ptom",
        "pho_sublead.r9",
        "pho_sublead.mvaID",
        "PV_log_score",
    ]
    return variable_list


def eval_dykiller_for_lowmass(diphotons, year="2022postEE"):
    model_dict = get_model_path()

    # add ptom, log PV_score
    diphotons["ptom"] = diphotons["pt"] / diphotons["mass"]
    diphotons["PV_log_score"] = np.log(diphotons["PV_score"])
    # model input variables
    variable_list = get_variable_list()
    dict_inputs = {
        var: ak.to_numpy(diphotons[tuple(var.split(".")) if "." in var else var])
        for var in variable_list
    }

    df_inputs = pd.DataFrame(dict_inputs)
    # ! if no events, just return
    if len(diphotons) < 1:
        # diphotons["dykiller"] = ak.zeros_like(diphotons.pt)
        return diphotons

    # ! Input df_inputs should not contain infinity or a value too large for dtype('float32')
    df_inputs = df_inputs.clip(
        np.finfo(np.float32).min + 1, np.finfo(np.float32).max - 1
    )

    # ! NN score
    ort_session = onnxruntime.InferenceSession(f"{model_dict[year]}")
    input_name = ort_session.get_inputs()[0].name

    # * eval NN
    predictions = ort_session.run(
        None, {input_name: df_inputs.to_numpy(dtype=np.float32)}
    )
    # sigmoid function
    preds = 1 / (1 + np.exp(-predictions[0]))

    # add dykiller score
    diphotons["dykiller_nn"] = ak.Array(ak.flatten(preds))

    return diphotons
