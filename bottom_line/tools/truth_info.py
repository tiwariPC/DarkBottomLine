import re
from typing import Dict


def get_truth_info_dict(filename: str) -> Dict[str, float]:

    # Get the dictonary of parameters and their values searching in the filename

    param_values = {}

    patterns = {
        "kl": r"_kl[-_](m?[\d]+p?[\d]*)",
        "kt": r"_kt[-_](m?[\d]+p?[\d]*)",
        "kX": r"_kX[-_](m?[\d]+p?[\d]*)",
        "ktX": r"_ktX[-_](m?[\d]+p?[\d]*)",
        "c2": r"_c2[-_](m?[\d]+p?[\d]*)",
        "CV": r"_CV[-_](m?[\d]+p?[\d]*)",
        "C2V": r"_C2V[-_](m?[\d]+p?[\d]*)",
        "C3": r"_C3[-_](m?[\d]+p?[\d]*)",
        "Radion_M": r"RadiontoHHto2B2G_M-([\d]+)",
        "BulkGraviton_M": r"BulkGravitontoHHto2B2G_M-([\d]+)",
        "XtoYHto2B2G_MX": r"NMSSM_XtoYHto2B2G_MX-([\d]+)",
        "XtoYHto2B2G_MY": r"NMSSM_XtoYHto2B2G.*_MY-([\d]+)",
        "XtoYHto2G2B_MX": r"NMSSM_XtoYHto2G2B_MX-([\d]+)",
        "XtoYHto2G2B_MY": r"NMSSM_XtoYHto2G2B.*_MY-([\d]+)",
        "rextriangle": r"_restriangle[-_](m?[\d]+p?[\d]*)",
        "resbox": r"_resbox[-_](m?[\d]+p?[\d]*)",
        "nonresonly": r"_nonresonly[-_](m?[\d]+p?[\d]*)",
    }
    # mY can be added later when the sample requests are submitted

    # Remove the {root_file}.root from the filename as it can contain kl,c2... patterns
    filename = filename.replace(filename.split("/")[-1], "")

    for key, pattern in patterns.items():
        match = re.search(pattern, filename)
        if match:
            param_values[key] = float(match.group(1).replace("m", "-").replace("p", "."))
    return param_values
