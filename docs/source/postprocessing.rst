Postprocessing
============================================

Standard Procedure
------------------

The standard way to get HiggsDNA Ntuples and transform them in FinalFit friendly output is to use the ``prepare_output_file.py`` script, provided and maintained in the ``script`` repository.
The script will perform multiple steps:
* Merge all the ``.parquet`` files and categorise the events, obtaining one file for each category of each sample.
* Convert the ``merged.parquet`` into ``ROOT`` trees.
* Convert the ``ROOT`` trees into FinalFit compatible ``RooWorkspace``s.

All the steps can be performed in one go with a command more or less like this::

        python3 prepare_output_file.py --input [path to output dir] --merge --root --ws --syst --cats --args "--do-syst"

or the single steps can be performed by running the auxiliary files (``merge_parquet.py``, ``convert_parquet_to_root.py``, ``Tree2WS``) separately.
A complete set of options for the main script is listed below.

Merging step
------------
During this step the main script calls ``merge_parquet.py`` multiple times. The starting point is the output of HiggsDNA, i.e. ``out_dir/sample_n/``. These directory **must** contain only ``.parquet`` files that have to be merged. 
The script will create a new directory called ``merged`` under ``out_dir``, if this directory already exists it will throw an error and exit.
When converting the data (in my case they were split per era, ``Data_B_2017``, ``Data_C_2017`` etc.) the script will put them in a new directory ``Data_2017`` and then merge again the output in a ``.parquet`` called ``allData_2017.parquet``.
During this step the events are also split into categories according to the boundaries defined in the ``cat_dict`` in the main file. An example of such dictionary is presented here::

        if opt.cats:
        cat_dict = {
            "best_resolution": {
                "cat_filter": [
                    ("sigma_m_over_m_decorr", "<", 0.005),
                    ("lead_mvaID", ">", 0.43),
                    ("sublead_mvaID", ">", 0.43),
                ]
            },
            "medium_resolution": {
                "cat_filter": [
                    ("sigma_m_over_m_decorr", ">", 0.005),
                    ("sigma_m_over_m_decorr", "<", 0.008),
                    ("lead_mvaID", ">", 0.43),
                    ("sublead_mvaID", ">", 0.43),
                ]
            },
            "worst_resolution": {
                "cat_filter": [
                    ("sigma_m_over_m_decorr", ">", 0.008),
                    ("lead_mvaID", ">", 0.43),
                    ("sublead_mvaID", ">", 0.43),
                ]
            },
        }

if you don't provide the dictionary to the script all the events will be put in a single file labelled as ``UNTAGGED``.

During the merging step MC samples can also be normalised to the ``efficiency x acceptance`` value as required later on by FinalFits, this step can be skipped using the tag ``--skip-normalisation``.

Root step 
---------

During this step the script calls multiple times the script ``convert_parquet_to_root.py``. The arguments to pass to the script, for instance if you want the systematic variation included in the output ``ROOT tree`` are specified when calling ``prepare_output_file.py`` using ``--args "--do-syst"``.
As before the script creates a new called ``root`` under ``out_dir``, if this directory already exists it will throw an error and exit. In the script there is a dictionary called ``outfiles`` that contains the name of the output root file that will be created according to the process tipe, if the wf is run using the main script this correspond to the proces containd in ``process_dict``.

By default, ``prepare_output_file.py`` uses the local execution to process files. If one wants to process the files via HTCondor (tested on LXPLUS), the ``--apptainer`` flag is to be used. It uses a docker image of the HiggsDNA master branch in conjunction with HTCondor to facilitate the work.

Data processing with local
--------------------------

To process the data locally, we have to know some things. First, we need to specify the absolute input path (``--input``) which leads to your output of ``run_analysis.py`` (unmerged parquet files). The output folder in which the merged parquet files are stored needs to be specified with ``--output``. If one wants to categorize the files, the ``--cats`` keyword is used in conjunction with ``--catsDict`` which points to the ``category.json`` to be considered. Are systematics desired, they have to be activated with ``--syst``.

In order to merge the parquet files according to the categories and produce the ROOT files in the same step, the following command is to be used:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input/path --cats --catDict /absolute/path/to/cat_data.json --varDict /absolute/path/to/varDict_data.json --syst --merge --root --output /absolute/output/path

Using the condor-way, one has to pay attention when processing data as an additional step wrt. the local-way is required, and the merge and ROOT-production step have to be separated:

Data processing with Docker
---------------------------

The first step is to merge the data parquet files according to the chosen categories. Since the data come in so-called eras (era ``A``, era ``B``, etc.), they have to be merged, such that we have per era and category a parquet file. This is the purpose of the following command, which has to be executed first:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input/path --cats --catDict /absolute/path/to/cat_data.json --varDict /absolute/path/to/varDict_data.json --syst --merge --output /absolute/output/path --apptainer

Studies in the past showed that for 2022 data there is not much of a difference significance-wise between splitting ``preEE`` and ``postEE`` datasets (referencing to the ECAL Endcap water leak in 2022) and merging them. For this reason, it was merged to one big dataset for HIG-23-014. The following command merges the era datasets to an ``allData.parquet`` file according to the categories. One needs in addition the flag ``--merge-data-only``:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input/path --cats --catDict /absolute/path/to/cat_data.json --varDict /absolute/path/to/varDict_data.json --syst --merge --output /absolute/output/path --merge-data-only --apptainer

Finally, we convert the parquet files to ROOT:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input_path/to_folder_with_merged --cats --catDict /absolute/path/to/cat_data.json --varDict /absolute/path/to/varDict_data.json --syst --root --output /absolute/input_path/to_folder_with_merged --apptainer

Whenever the parquet files are merged (after the first step), a folder ``merged`` in the ``/absolute/output/path`` is created. For getting the ROOT files, one has to use the folder ``/absolute/output/path`` (which is now containing the ``merged`` subfolders) as the new input folder. The file processing for MC samples functions in a similar way:

MC processing with Docker
-------------------------

Similar to data, the MC samples can be processed with HTCondor. Here we only have two steps. The first consists of merging the parquet files according to the categories just like in the data case:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input/path --cats --catDict /absolute/path/to/cat_mc.json --varDict /absolute/path/to/varDict_mc.json --syst --merge --output /absolute/output/path --apptainer

In order to convert the parquet files to ROOT, one executes:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input_path/to_folder_with_merged --cats --catDict /absolute/path/to/cat_mc.json --varDict /absolute/path/to/varDict_mc.json --syst --root --output /absolute/input_path/to_folder_with_merged --apptainer

One can specify a separate path which is hosting all the sub and sh files with ``--condor-logs``. If the condor log, err, and out files are desired (e.g. for debugging purposes) they can be explicitly produced with ``--make-condor-logs``.

A valid command would for example be:

.. code-block:: python

    python prepare_output_file.py --input /absolute/input/path --cats --catDict /absolute/path/to/cat_mc.json --varDict /absolute/path/to/varDict_mc.json --syst --merge --output /absolute/output/path --condor-logs /absolute/path/to/condor/logs --make-condor-logs --apptainer


Workspace step
--------------

During this step the main script uses multiple time the ``Flashgg_FinalFit``, it moves to the directory defined in the ``--final-fit`` option (improvable) and uses the ``Tree2WS`` script there on the content of the ``root`` directory previously created. The output is stored in ``out_dir/root/smaple_name/ws/``.

Commands
--------

The workflow is meant to be run in one go using the ``prepare_output_file.py`` script, it can be also split in different steps or run with the single auxiliary files but it can result a bit cumbersome.

To run everything starting from the output of HiggsDNA with categories and systematic variatrion one can use::

        python3 prepare_output_file.py --input [path to output dir] --merge --root --ws --syst --cats --args "--do-syst"

and everithing should run smoothly, it does for me at least (I've not tried the scripts in a while so thing may have to be adjusted in this document).
Some options can be removed. If you want to use ``--syst`` and ``--root`` you should also add ``--args "--do-syst"``.

The complete list of options for the main file is here:

    * ``--merge``, "Do merging of the .parquet files"
    * ``--root``, "Do root conversion step"
    * ``--ws``, "Do root to workspace conversion step"
    * ``--ws-config``, "configuration file for Tree2WS, as it is now it must be stored in Tree2WS directory in FinalFit",
    * ``--final-fit``, "FlashggFinalFit path" # the default is just for me, it should be changed but I don't see a way to make this generally valid
    * ``--syst``, "Do systematics variation treatment"
    * ``--cats``, ="Split into categories",
    * ``--args``, "additional options for root converter: --do-syst, --notag",
    * ``--skip-normalisation``, "Independent of file type, skip normalisation step",
    * ``--verbose``, "verbose lefer for the logger: INFO (default), DEBUG",
    * ``--output``, "Output path for the merged and ROOT files.",
    * ``--folder-structure``, "Uses the given folder structure for the dirlist. Mainly used for debug purposes.",
    * ``--apptainer``, "Run HTCondor with Docker image of HiggsDNA's current master branch.",
    * ``--merge-data-only``, "Flag for merging data to an allData file. Only used when --condor is used, and only when we process data.",
    * ``--make-condor-logs``, "Create condor log files.",
    * ``--condor-logs``, "Output path of the Condor log files.",


The merging step can also be run separately using::

        python3 merge_parquet.py --source [path to the directory containing .parquets] --target [target directory path] --cats [cat_dict]

the script works also without the ``--cats`` option, it creates a dummy selection of ``Pt > -1`` and call the category ``UNTAGGED``.

Same for the root step::

        python3 convert_parquet_to_root.py [/path/to/merged.parquet] [path to output file containing also the filename] mc (or data depending what you're doing) --process [process name (should match one of the outfiles dict entries)] --do-syst --cats [cat_dict] --vars [variation.json]

``--do-syst`` is not mandatory, but if it's there also the dictionary containing the variations must be specified with the ``--var`` option. As before the script works also without the ``--cats`` option.



