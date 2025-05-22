=============
Main Concepts
=============


.. _def-cltool:

-----------------
Command Line Tool
-----------------
If you want to run an analysis with processors and taggers that have already been developed, the suggested way is to use the command line tool ``run_analysis.py``.

The main part that define an analysis are the following:

* ``datasets``:
  path to a JSON file in the form ``{"dataset_name": [list_of_files]}`` (like the one dumped by dasgoclient)
* ``workflow``:
  the coffea processor you want to use to process you data, can be found in the modules located inside the subpackage ``bottom_line.workflows``
* ``metaconditions``:
  the name (without ``.json`` extension) of one of the JSON files inherited from FLASHgg and located inside ``bottom_line.metaconditions``
* ``year``: the year condition for each sample, let us use ``2016preVFP``, ``2016postVFP``, ``2017``, ``2018``, ``2022preEE``, ``2022postEE``, ``2023``
* ``taggers``:
  the set of taggers you want to use, can be found in the modules located inside the subpackage ``bottom_line/workflows/taggers``
* ``systematics``: the set of systematics you want to use for each sample
* ``corections``: the set of corrections you want to use for each sample

These parameters are specified in a JSON file and passed to the command line with the flag ``--json-analysis``

.. code-block:: bash

        run_analysis.py --json-analysis simple_analysis.json

where ``simple_analysis.json`` looks like this:

.. code-block:: json

      {
          "samplejson": "/work/gallim/devel/HiggsDNA/tmp/DY-data-test.json",
          "workflow": "tagandprobe",
          "metaconditions": "Era2017_legacy_xgb_v1",
          "taggers": [
              "DummyTagger1"
          ],
          "year":[
            "SampleName1": ["2022preEE"],
            "SampleName2": ["2017"]
          ]
          "systematics": {
              "SampleName1": [
                  "SystematicA", "SystematicB"
              ],
              "SampleName2": [
                  "SystematicC"
              ]
          },
          "corrections": {
                  "SampleName1": [
                      "CorrectionA", "CorrectionB"
                  ],
                  "SampleName2": [
                      "CorrectionC"
                  ]
              }
      }

where the ``taggers`` list and the ``systematics`` or ``corrections`` dictionaries can be left empty if no taggers or systematics are applied. For most of the ``systematics``, a ``year`` condition is necessary.


The next two flags that you will want to specify are ``dump`` and ``executor``: the former receives the path to a directory where the parquet output files will be stored, while the latter specifies the Coffea executor used to process the chunks of data. It can be one of the following:

* ``iterative``
* ``futures``
* ``dask/local``
* ``dask/condor``
* ``dask/slurm``
* ``dask/lpc``
* ``dask/lxplus``
* ``dask/casa``
* ``parsl/slurm``
* ``parsl/condor``

There are then a few other options that depend on the backend. When running with Dask, for instance, you may want to use the command line to change the following parameters:

* ``workers``:
  number processes/threads used **on the same node** (this means that every job will use this amount of cores)
* ``scaleout``:
  minimum number of nodes to scale out to (i.e. minimum number of jobs submitted)
* ``max-scaleout``:
  maximum number of nodes to adapt your cluster to (i.e. maximum number of jobs submitted)

As usual, a description of all the options is printed when running::

        run_analysis.py --help

.. _def-processor:

----------
Processors
----------
Processors are items defined within Coffea where the analysis workflow is described. While a general overview is available in the `Coffea documentation <https://coffeateam.github.io/coffea/concepts.html#coffea-processor>`_, here we will focus on the aspects that are important for HiggsDNA.

Since in Higgs to diphoton analysis there are some operations that are common to every analysis workflow, we wrote a base processor `bbMETBaseProcessor <https://higgs-dna.readthedocs.io/en/latest/modules/bottom_line.workflows.html#bottom_line.workflows.base.bbMETBaseProcessor>`_ which can be used in many basic analyses. If more complex operations are needed, one can still write a processor that inherits from the base class and redefines the function ``process``. The operations that one can find within ``bbMETBaseProcessor.process`` are the following:

* application of filters and triggers
* Chained Quantile Regression to correct shower shapes and isolation variables
* photon IdMVA
* diphoton IdMVA
* photon preselection
* event tagging
* application of systematic uncertainties

Write a New Processor
---------------------

There are cases in which the workflows implemented in HiggsDNA are not enough for your studies. In these cases you might need to **write your own processor**. Depending on the scenario, there are different guidelines to do this.

1. **Hgg-like workflow**. In this case your analysis is similar to the one implemented in the Hgg basic processor, but you need to perform other operations on top (e.g. additional cuts, application of NNs, etc.). In order to reduce the amount **repeated code**, what you can do is write a processor that inherits from ``bbMETBaseProcessor`` and redefine the function ``process_extra``. You can find an example of this in `DYStudiesProcessor <https://higgs-dna.readthedocs.io/en/latest/modules/bottom_line.workflows.html#bottom_line.workflows.dystudies.DYStudiesProcessor>`_.

2. **Non Hgg-like workflow**. This is the case in which the operations you need to perform are different from the ones performed in the ``process`` function of ``HggBaseProcess``. In this kind of scenario you can still inherit from ``bbMETBaseProcessor`` in order to have access to the same attributes, but you also need to rewrite the ``process`` function. An example of this is the `TagAndProbeProcessor <https://higgs-dna.readthedocs.io/en/latest/_modules/bottom_line/workflows/dystudies.html#TagAndProbeProcessor>`_. In this case, we cannot use the standard workflow since we manipulate objects in a different way (for instance, we have *tag* and *probe* photons instead of lead and sublead and since each item of a pair can be either tag or probe we need to double the number of candidates - this is an operation that we would never do in a standard workflow).

-------
Taggers
-------

------------------------
Systematic Uncertainties
------------------------
