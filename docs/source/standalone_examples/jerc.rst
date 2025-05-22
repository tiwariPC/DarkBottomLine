.. default-role:: math

JERC and systematics
=============

The ``JEC`` and ``JER`` could be considered via the `correctionlib <https://github.com/cms-nanoAOD/correctionlib>`_, which read `JSON files <https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME?ref_type=heads>`_ of latest ``Jet Energy Scale``, ``Jet Energy Resolution``, and ``Jet Veto Maps``.

The ``JEC`` and ``JER`` correct the 4-momentum of each jet. Thus, both jet *p\T* and *mass* are corrected. So it's not easy to applied the ``JERC`` corrections and systematics seperately as most of the other corrections and systematics. Also, if ``JER`` is considered, the ``JEC`` systematics should be derived based on the ``JER`` corrected jets.

A series of the functions are provided. Some of them also derive **systematics**, so the branches contain systematic variations are added to the jets collection. Then, function ``get_obj_syst_dict`` in `dumping_utils.py <https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/bottom_line/utils/dumping_utils.py?ref_type=heads>`_ splits the nominal and variations into the jet systematic dictionary, which has the same structure with the ``photons_dct`` in `base.py <https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/bottom_line/workflows/base.py?ref_type=heads>`_.

The correction functions and their goals are as following:

* ``jec_jet``: redo the **Jet Energy Scale Correction**

* ``jec_jet_syst``: redo the **Jet Energy Scale Correction** and add **total** up/down systematics

* ``jec_jet_regrouped_syst``: redo the **Jet Energy Scale Correction** and add up/down systematics of each **Regrouped** source

* ``jerc_jet``: redo the **Jet Energy Scale Correction** and apply the **Jet Energy Resolution Smearing**

* ``jerc_jet_syst``: redo the **Jet Energy Scale Correction** and apply the **Jet Energy Resolution Smearing**. Also add **total** up/down ``JEC`` systematics and ``JER`` systematics

* ``jerc_jet_regrouped_syst``: redo the **Jet Energy Scale Correction** and apply the **Jet Energy Resolution Smearing**. Also add up/down ``JEC`` systematics of each **Regrouped** source and ``JER`` systematics

* For ``data``, only redo **Jet Energy Scale Correction** is allowed. But the ``JEC`` for data need to consider the ``Era``, so please use the correct ``Era`` for the data sample from the following correction functions

    - ``jec_RunA``
    - ``jec_RunB``
    - ``jec_RunC``
    - ``jec_RunD``
    - ``jec_RunE``
    - ``jec_RunF``
    - ``jec_RunG``
    - ``jec_RunH``

An example to do ``JEC`` and ``JER`` with considering their systematics is provided here. The configuration json ``test_run3_v0.json`` is as following:

.. code-block:: json

    {
        "samplejson": "samples_nanov12_EE_v0.json",
        "workflow": "base",
        "metaconditions": "Era2022_v1",
        "taggers": [],
        "year": {
            "DYto2L_EE": [
                "2022postEE"
            ],
            "EGammaF": [
                "2022postEE"
            ]
        },
        "systematics": {
        },
        "corrections": {
            "DYto2L_EE": [
                "jerc_jet_syst","Smearing"
            ],
            "EGammaF": [
                "jec_RunF"
            ]
        }
    }

The sample json file ``samples_nanov12_EE_v0.json`` looks like

.. code-block:: json

    {
        "DYto2L_EE": [
            "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/Run3Summer22EENanoAODv12/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/50000/fa9b6dc8-06e7-46aa-a936-ca60aa8af867.root"
        ],
        "EGammaF":[
            "/eos/lyoeos.in2p3.fr/grid/cms/store/data/Run2022F/EGamma/NANOAOD/22Sep2023-v1/50000/2ada79a4-a7ce-4f4a-ad19-bd5d436a53c5.root"
        ]
    }

With running the command

.. code-block:: bash

    python ../scripts/run_analysis.py --dump output12 --json-analysis test_run3_v0.json --save nanov12.coffea --no-trigger --debug


The output files are in such a structure

.. code-block:: bash

    output12/
    |-- DYto2L_EE
    |   |-- jec_syst_Total_down
    |   |   `-- 02b4758c-61dc-11ee-b6b1-ae013c0abeef_%2FEvents%3B1_0-695760.parquet
    |   |-- jec_syst_Total_up
    |   |   `-- 02b4758c-61dc-11ee-b6b1-ae013c0abeef_%2FEvents%3B1_0-695760.parquet
    |   |-- jer_syst_down
    |   |   `-- 02b4758c-61dc-11ee-b6b1-ae013c0abeef_%2FEvents%3B1_0-695760.parquet
    |   |-- jer_syst_up
    |   |   `-- 02b4758c-61dc-11ee-b6b1-ae013c0abeef_%2FEvents%3B1_0-695760.parquet
    |   `-- nominal
    |       `-- 02b4758c-61dc-11ee-b6b1-ae013c0abeef_%2FEvents%3B1_0-695760.parquet
    `-- EGammaF
        `-- nominal
            |-- f128d04c-5a97-11ee-8144-8bd09a83beef_%2FEvents%3B1_0-463938.parquet
            `-- f128d04c-5a97-11ee-8144-8bd09a83beef_%2FEvents%3B1_463938-927875.parquet

--------------------------------
Run3 splitted JEC systematics
--------------------------------

Now, for Run3, only ``total`` uncertainty or ``full splitted JEC systematics`` available, ``Regrouped`` uncertainties are in development. The function  ``jec_jet_regrouped_syst`` in fact give the ``full splitted JEC systematics``. An example json configuration is

.. code-block:: json

    {
        "samplejson": "samples_nanov12_dy_v0.json",
        "workflow": "base",
        "metaconditions": "Era2022_v1",
        "taggers": [],
        "year": {
            "DYto2L_EE": [
                "2022postEE"
            ]
        },
        "systematics": {},
        "corrections": {
            "DYto2L_EE": [
                "jec_jet_regrouped_syst"
            ]
        }
    }

The related sample json is

.. code-block:: json

    {
        "DYto2L_EE": [
            "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/Run3Summer22EENanoAODv12/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/50000/d2b250b2-8886-4dcb-b20b-4e6c89a9dfa4.root"
        ]
    }

The output files is in the following structure, each source has one dedicated folder

.. code-block:: bash

    output_dl/DYto2L_EE/
    |-- jec_syst_AbsoluteMPFBias_down
    |-- jec_syst_AbsoluteMPFBias_up
    |-- jec_syst_AbsoluteScale_down
    |-- jec_syst_AbsoluteScale_up
    |-- jec_syst_AbsoluteStat_down
    |-- jec_syst_AbsoluteStat_up
    |-- jec_syst_FlavorQCD_down
    |-- jec_syst_FlavorQCD_up
    |-- jec_syst_Fragmentation_down
    |-- jec_syst_Fragmentation_up
    |-- jec_syst_PileUpDataMC_down
    |-- jec_syst_PileUpDataMC_up
    |-- jec_syst_PileUpPtBB_down
    |-- jec_syst_PileUpPtBB_up
    |-- jec_syst_PileUpPtEC1_down
    |-- jec_syst_PileUpPtEC1_up
    |-- jec_syst_PileUpPtEC2_down
    |-- jec_syst_PileUpPtEC2_up
    |-- jec_syst_PileUpPtHF_down
    |-- jec_syst_PileUpPtHF_up
    |-- jec_syst_PileUpPtRef_down
    |-- jec_syst_PileUpPtRef_up
    |-- jec_syst_RelativeBal_down
    |-- jec_syst_RelativeBal_up
    |-- jec_syst_RelativeFSR_down
    |-- jec_syst_RelativeFSR_up
    |-- jec_syst_RelativeJEREC1_down
    |-- jec_syst_RelativeJEREC1_up
    |-- jec_syst_RelativeJEREC2_down
    |-- jec_syst_RelativeJEREC2_up
    |-- jec_syst_RelativeJERHF_down
    |-- jec_syst_RelativeJERHF_up
    |-- jec_syst_RelativePtBB_down
    |-- jec_syst_RelativePtBB_up
    |-- jec_syst_RelativePtEC1_down
    |-- jec_syst_RelativePtEC1_up
    |-- jec_syst_RelativePtEC2_down
    |-- jec_syst_RelativePtEC2_up
    |-- jec_syst_RelativePtHF_down
    |-- jec_syst_RelativePtHF_up
    |-- jec_syst_RelativeSample_down
    |-- jec_syst_RelativeSample_up
    |-- jec_syst_RelativeStatEC_down
    |-- jec_syst_RelativeStatEC_up
    |-- jec_syst_RelativeStatFSR_down
    |-- jec_syst_RelativeStatFSR_up
    |-- jec_syst_RelativeStatHF_down
    |-- jec_syst_RelativeStatHF_up
    |-- jec_syst_SinglePionECAL_down
    |-- jec_syst_SinglePionECAL_up
    |-- jec_syst_SinglePionHCAL_down
    |-- jec_syst_SinglePionHCAL_up
    |-- jec_syst_TimePtEta_down
    |-- jec_syst_TimePtEta_up
    |-- jec_syst_Total_down
    |-- jec_syst_Total_up
    `-- nominal

--------------------------------
Run2UL Regrouped JEC systematics
--------------------------------

For Run2UL, to consider ``Regrouped`` uncertainties of JEC. The function ``jerc_jet_regrouped_syst`` could be used.

The example json configuration is

.. code-block:: json

    {
        "samplejson": "samples_nanov9_v0.json",
        "workflow": "base",
        "metaconditions": "Era2018_legacy_v1",
        "taggers": [],
        "year": {
            "DYto2L": [
                "2018"
            ]
        },
        "systematics": {
        },
        "corrections": {
            "DYto2L": [
                "jerc_jet_regrouped_syst"
            ]
        }
    }

The related sample json is

.. code-block:: json

    {
        "DYto2L": [
            "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/8559CF2F-9B52-3A4D-9780-5499A2751135.root"
        ]
    }

The output files is in the following structure, each **Regrouped** source has one dedicated folder

.. code-block:: bash

    output9/
    `-- DYto2L
        |-- jec_syst_Absolute_2018_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_Absolute_2018_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_Absolute_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_Absolute_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_BBEC1_2018_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_BBEC1_2018_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_BBEC1_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_BBEC1_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_EC2_2018_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_EC2_2018_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_EC2_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_EC2_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_FlavorQCD_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_FlavorQCD_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_HF_2018_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_HF_2018_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_HF_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_HF_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_Regrouped_Total_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_Regrouped_Total_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_RelativeBal_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_RelativeBal_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_RelativeSample_2018_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jec_syst_RelativeSample_2018_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jer_syst_down
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        |-- jer_syst_up
        |   |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
        |   `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet
        `-- nominal
            |-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_0-468807.parquet
            `-- ce6b7240-1c7c-11ec-85e3-5c090d0abeef_%2FEvents%3B1_468807-937614.parquet