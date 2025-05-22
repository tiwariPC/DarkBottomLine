import awkward as ak
import torch
import zuko
import numpy as np
import sys
import os


def apply_flow_corrections_to_photons(photons, events, meta, year, add_photonid_mva_run3, logger=None):
    """
    Apply normalizing flow corrections to the photon collection and update the photonID MVA.

    Parameters:
        photons (ak.Array): The photon collection.
        events (ak.Array): The events collection.
        meta (dict): Metadata containing correction inputs".
        year (str): The dataset year identifier.
        add_photonid_mva_run3 (callable): Function to recompute the photonID MVA.
        logger (logging.Logger, optional).

    Returns:
        ak.Array: The updated photon collection with flow corrections applied.
    """
    if logger is not None:
        logger.info("Calculating normalizing flow corrections to photon MVA ID inputs")

    # Get counts to be used for unflattening the corrected inputs.
    counts = photons.num() if hasattr(photons, "num") else ak.num(photons)

    # Calculate corrected inputs
    corrected_inputs, var_list = calculate_flow_corrections(
        photons,
        events,
        meta["flashggPhotons"]["flow_inputs"],
        meta["flashggPhotons"]["Isolation_transform_order"],
        year=year
    )

    # Preserve the original photonID MVA before applying corrections.
    photons["mvaID_nano"] = photons["mvaID"]

    # For each variable, store the raw value and update with the corrected value.
    for i, var in enumerate(var_list):
        photons["raw_" + str(var)] = photons[str(var)]
        photons[str(var)] = ak.unflatten(corrected_inputs[:, i], counts)

    # Update the photonID MVA using the new, corrected inputs.
    photons["mvaID"] = ak.unflatten(add_photonid_mva_run3(photons, events), counts)

    return photons


# Function responsible for applyting the flow and calculate the per photon corrections in the mvaID inputs and in the sigma_E/E
def calculate_flow_corrections(photon: ak.Array, events, inputs_list, isolation_indexes , year="2022postEE"):

    """
    Perform the evaluation of pre-trained flow models for simulation to data corrections

    inputs:
    -> photon: Array containing each photon candidate information
    -> events: Contains event information, like pile-up, which is needed for the flow correction
    -> year:   Since we have pre and postEE samples, and they are ReReco and prompt, two models were trained for each period

    output -> array containing corrected entries and var_list
    """

    # We will correct mvaID input variables and energyErr (used for sigma_m correction)
    var_list = inputs_list

    # These variables will be used as conditions to the normalizing flow - they will not be transformed!
    # Rho and the IsData boolean is also used, but they will be added later, since the photon container dont have pile up information
    conditions_list = ["pt","ScEta","phi"]

    # Reading the normalizing flow models!
    if (year == "2022postEE"):
        flow = zuko.flows.NSF(len(var_list), context=len(conditions_list) + 2, bins=10, transforms=5, hidden_features=[256] * 2)
        path_means_std = os.path.join(os.path.dirname(__file__), 'flows/postEE/')
        flow.load_state_dict(torch.load(path_means_std + 'best_model_.pth', map_location=torch.device('cpu'), weights_only=False))
    elif (year == "2022preEE"):
        flow = zuko.flows.NSF(len(var_list), context=len(conditions_list) + 2, bins=10, transforms=5, hidden_features=[256] * 2)
        path_means_std = os.path.join(os.path.dirname(__file__), 'flows/preEE/')
        flow.load_state_dict(torch.load(path_means_std + 'best_model_.pth', map_location=torch.device('cpu'), weights_only=False))
    elif ('2023' in year):
        flow = zuko.flows.NSF(len(var_list), context=len(conditions_list) + 2, bins=10, transforms=5, hidden_features=[256] * 2, passes=2)
        path_means_std = os.path.join(os.path.dirname(__file__), 'flows/2023_model/')
        flow.load_state_dict(torch.load(path_means_std + 'best_model_.pth', map_location=torch.device('cpu'), weights_only=False))
    else:
        print('\nThere is no model trained for this specific year!! - Exiting')
        sys.exit(0)

    rho = events.Rho.fixedGridRhoAll * ak.ones_like(photon.pt)
    rho = ak.flatten(rho)
    photon = ak.flatten(photon)

    flow_inputs = {}
    flow_inputs = np.column_stack(
        [ak.to_numpy(photon[name]) for name in var_list]
    )

    flow_conditions = {}
    flow_conditions = np.column_stack(
        [ak.to_numpy(photon[name]) for name in conditions_list]
    )

    # Adding the boolean to the conditions - rho has to be added by hand here, since it is a event quantity, not a photon one.
    flow_conditions = np.concatenate([flow_conditions, np.array(rho).reshape(-1,1), np.zeros(len(photon)).reshape(-1,1)], axis=1)

    # numpy to pytorch
    flow_conditions = torch.tensor(flow_conditions)
    flow_inputs = torch.tensor(flow_inputs)

    # Performing the transformations in the inputs and conditions arrays
    input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc = perform_pre_processing(flow_inputs,flow_conditions,isolation_indexes , path_means_std)

    samples = apply_flow(input_tensor, conditions_tensor, flow)

    # Inverting the transformations!
    corrected_inputs = invert_pre_processing(samples, input_mean_for_std, input_std_for_std, isolation_indexes ,vector_for_iso_constructors_mc)

    return corrected_inputs.detach().cpu().numpy(), var_list


def apply_flow(input_tensor : torch.tensor, conditions_tensor : torch.tensor, flow) -> torch.tensor:

    """
    This function is responsable for applying the normalizing flow to MC samples

    Inputs:
    -> input_tensor: pytorch tensor (contaings the inputs distirbutions)
    -> conditions_tensor: torch.tensor
    -> flow (pre-trained zuko model to perform the simulation corrections)

    Output: pytorch tensor containing the flow corrected entries
    """

    # Making sure flow and input tensors have the same type
    flow = flow.type(input_tensor.dtype)
    conditions_tensor = conditions_tensor.type(input_tensor.dtype)

    # Use cuda if avaliable - maybe this is causing the meory problems?
    device = torch.device('cpu')
    flow = flow.to(device)
    input_tensor = input_tensor.to(device)
    conditions_tensor = conditions_tensor.to(device)

    # Disabling pytorch gradient calculation so operation uses less memory and is faster
    with torch.no_grad():

        # MC space to data space
        trans = flow(conditions_tensor).transform
        sim_latent = trans(input_tensor)

        # Flip of the IsData boolean
        conditions_tensor = torch.tensor(np.concatenate([conditions_tensor[:,:-1].cpu(), np.ones_like(conditions_tensor[:,0].cpu()).reshape(-1,1)], axis=1)).to(device)

        # from latent to data space
        trans2 = flow(conditions_tensor).transform
        samples = trans2.inv(sim_latent)

    return samples


def perform_pre_processing(input_tensor : torch.tensor, conditions_tensor : torch.tensor, isolation_indexes, path=False):

    # Fist we make the isolation variables transformations
    # The indexes_for_iso_transform arrays point to the indexes in the inputs where the isolation variables are
    indexes_for_iso_transform = isolation_indexes
    vector_for_iso_constructors_mc = []

    # creating the constructors
    for index in indexes_for_iso_transform:

        # since hoe has very low values, the shift value (value until traingular events are sampled) must be diferent here
        if (index == 6):
            vector_for_iso_constructors_mc.append(Make_iso_continuous(input_tensor[:,index], device=torch.device('cpu'), b=0.001))
        else:
            vector_for_iso_constructors_mc.append(Make_iso_continuous(input_tensor[:,index], device=torch.device('cpu')))

    # Applying the transformations
    counter = 0
    for index in indexes_for_iso_transform:

        # transforming the training dataset
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].shift_and_sample(input_tensor[:,index])
        counter = counter + 1

    # Now, the standartization -> The arrays means are the same ones using during training/evaluation
    input_mean_for_std = torch.tensor(np.load(path + 'input_means.npy'))
    input_std_for_std = torch.tensor(np.load(path + 'input_std.npy'))
    condition_mean_for_std = torch.tensor(np.load(path + 'conditions_means.npy'))
    condition_std_for_std = torch.tensor(np.load(path + 'conditions_std.npy'))

    # Standardizing!
    input_tensor = (input_tensor - input_mean_for_std) / input_std_for_std
    conditions_tensor[:,:-1] = (conditions_tensor[:,:-1] - condition_mean_for_std) / condition_std_for_std

    return input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc


# revert the corrected samples tranformation
def invert_pre_processing(input_tensor: torch.tensor, input_mean_for_std: torch.tensor, input_std_for_std: torch.tensor, isolation_indexes ,vector_for_iso_constructors_mc) -> torch.tensor:

    indexes_for_iso_transform = isolation_indexes

    # inverting the standartization
    input_tensor = (input_tensor * input_std_for_std + input_mean_for_std)

    # Now inverting the isolation transformation
    counter = 0
    for index in indexes_for_iso_transform:

        # now transforming the
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(input_tensor[:,index], processed=True)
        counter = counter + 1
    return input_tensor


class Make_iso_continuous:
    def __init__(self, tensor, device, b=False):

        self.device = device
        self.iso_bigger_zero = tensor > 0
        self.iso_equal_zero = tensor == 0
        self.shift = 0.05
        if (b):
            self.shift = b
        self.n_zero_events = torch.sum(self.iso_equal_zero)
        self.before_transform = tensor.clone().detach()

    # Shift the continous part of the continous distribution to (self.shift), and then sample values for the discontinous part
    def shift_and_sample(self, tensor):

        # defining two masks to keep track of the events in the 0 peak and at the continous tails
        bigger_than_zero = tensor > 0
        tensor_zero = tensor == 0
        self.lowest_iso_value = 0

        tensor[bigger_than_zero] = tensor[bigger_than_zero] + self.shift - self.lowest_iso_value
        tensor[tensor_zero] = torch.tensor(np.random.triangular(left=0., mode=0, right=self.shift * 0.99, size=tensor[tensor_zero].size()[0]), dtype=tensor[tensor_zero].dtype)

        # now a log trasform is applied on top of the smoothing to stretch the events in the 0 traingular and "kill" the iso tails
        tensor = torch.log(1e-3 + tensor)

        return tensor.to(self.device)

    # inverse operation of the above shift_and_sample transform
    def inverse_shift_and_sample(self,tensor, processed=False):
        tensor = torch.exp(tensor) - 1e-3

        bigger_than_shift = tensor > self.shift
        lower_than_shift = tensor < self.shift

        tensor[lower_than_shift] = 0
        tensor[bigger_than_shift] = tensor[bigger_than_shift] - self.shift

        return tensor.to(self.device)
