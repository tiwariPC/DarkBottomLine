Base workflow
=============

In our language, the base workflow refers to a standard :math:`H \rightarrow \gamma \gamma` NTuplisation operation. This includes a CMS diphoton preselection and writing out the relevant variables of the two leading photons. Systematics can be added as desired (and currently implemented).
The processor that implements the main operations performed can be found at ``bottom_line/workflows/base:bbMETBaseProcessor``.

The first ingredient is a "sample JSON" file that includes the datasets with their corresponding files.
Both local files and file access via xrootd is supported.
The current master of HiggsDNA works with v11 nanoAODs, so let's specify a v11 GJet nanoAOD as an example.
You can try to specify additional files from the sample if you want to later. For convenience, we give the `DAS link <https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D/GJet_PT-20_DoubleEMEnriched_MGG-40to80_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv11-126X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM>`_.

.. code-block:: json

    {
        "GJet": [
            "root://cms-xrd-global.cern.ch//store/mc/Run3Summer22EENanoAODv11/GJet_PT-20_DoubleEMEnriched_MGG-40to80_TuneCP5_13p6TeV_pythia8/NANOAODSIM/126X_mcRun3_2022_realistic_postEE_v1-v2/2550000/0cba4864-da82-4a1e-adfb-959a20b1970a.root"
        ]
    }

Save this as a file ``sampleJSON.json``, for example in the root directory of HiggsDNA (but the concrete location is not important).

We have to specify a so-called "runner JSON" that encodes the conditions of the HiggsDNA run.
There, we give the path to the Sample JSON file (see paragraph above), we define the workflow, the metaconditions, any taggers that should be used as well as systematics and corrections (in this example, we choose to add scale factors for the photon ID, currently a dummy systematic to show how it looks in the output...).
We save this file as ``runnerJSON.json`` (for example in the root directory of HiggsDNA).

.. code-block:: json

    {
        "samplejson": "<path_to_sampleJSON.json>",
        "workflow": "base",
        "metaconditions": "Era2017_legacy_xgb_v1",
        "taggers": [],
        "systematics": {
            "GJet": ["SF_photon_ID"]
        },
        "corrections": {
            "GJet": ["SF_photon_ID"]
        }
    }

Finally, we have the runner command:

.. code-block:: bash

   python <path_to_run_analysis.py> --json-analysis <path_to_runnerJSON.json> --dump <path_to_dump>

The argument ``<path_to_dump>`` refers to a writable path on your system and will contain the output of HiggsDNA.
The directory will be created and does not need to present prior to executing the command.

Please note that the output is currently given in parquet files.
In ``scripts``, we provide helper scripts to help you convert the output into ROOT files so that you can investigate them quickly in a TBrowser or interactively in a ROOT session.