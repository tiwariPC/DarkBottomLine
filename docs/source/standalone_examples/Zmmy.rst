.. default-role:: math

`Z(\mu \mu \gamma)` processor
=============

The `Z(\mu \mu \gamma)` processors test the single photon from `Z \rightarrow \mu \mu` final state radiation (FSR). There is a  processor for ntuplizing ``ZmmyProcessor``, a processor for generating histograms ``ZmmyHist``, and a processor ``ZmmyZptHist`` for generating ``Zpt`` distribution to derive ``Zpt`` reweighting.

An example ``json`` configuration is as following

.. code-block:: json

    {
        "samplejson": "samples_run3_dy_v0.json",
        "workflow": "zmmy",
        "metaconditions": "Era2022_v1",
        "taggers": [],
        "year": {
            "DYto2L_EE": [
                "2022postEE"
            ],
            "MuonF": [
                "2022postEE"
            ],
            "MuonG": [
                "2022postEE"
            ]
        },
        "systematics": {
            "DYto2L_EE": [
                "Pileup"
            ]
        },
        "corrections": {
            "DYto2L_EE": [
                "Pileup",
                "Smearing"
            ],
            "MuonF": [
                "Scale"
            ],
            "MuonG": [
                "Scale"
            ]
        }
    }

To run the configuration, use command

.. code-block:: bash

    python ../scripts/run_analysis.py --json-analysis test_run3_mmy_dy_v0.json --dump output0 --executor futures --save count.coffea

To make histograms, let us start with the ``Zpt`` distribution. The ``json`` configuration is

.. code-block:: json

    {
        "samplejson": "samples_mmy_hist.json",
        "workflow": "zmmyZptHist",
        "metaconditions": "Era2022_v1",
        "taggers": [],
        "year": {
            "DYto2L_EE": [
                "2022postEE"
            ],
            "MuonF": [
                "2022postEE"
            ]
        },
        "systematics": {},
        "corrections": {}
    }

The sample json file ``samples_mmy_hist.json`` looks like

.. code-block:: json

    {
        "DYto2L_EE": [
            "output0/DYto2L_EE/441a0620-626f-11ee-8f7f-3abfe183beef_%2FEvents%3B1_0-459437.parquet",
            "output0/DYto2L_EE/441a0620-626f-11ee-8f7f-3abfe183beef_%2FEvents%3B1_459437-918874.parquet"
        ],
        "MuonF": [
            "output0/MuonF/572e9c40-5d05-11ee-9aa8-7f013c0abeef_%2FEvents%3B1_0-50166.parquet",
            "output0/MuonF/91b32364-5d09-11ee-8941-a60e010abeef_%2FEvents%3B1_0-600832.parquet"
        ]
    }

The running command is

.. code-block:: bash

    python ../scripts/run_analysis.py --json-analysis test_hist_Zpt.json --schema base --format parquet --save hist_Zpt.coffea

Example code to generate ``Zpt`` reweighting json file:

.. code-block:: bash
    python ../script/Zpt_rwgt_json.py --norm count.coffea --hist hist_Zpt.coffea -o my_Zpt_reweighting.json.gz

Example configuration to make more plots with considering ``Zpt`` reweighting. Here, the ``ZmmyHist`` processor (alias as ``zmmyHist``) is used

.. code-block:: json

    {
        "samplejson": "samples_mmy_hist.json",
        "workflow": "zmmyHist",
        "metaconditions": "Era2022_v1",
        "taggers": [],
        "year": {
            "DYto2L_EE": [
                "2022postEE"
            ],
            "MuonF": [
                "2022postEE"
            ]
        },
        "systematics": {},
        "corrections": {
            "DYto2L_EE": [
                "Zpt"
            ]
        }
    }

The running command is,

.. code-block:: bash

    python ../scripts/run_analysis.py --json-analysis test_hist.json --schema base --format parquet --save hist_Zmmy.coffea