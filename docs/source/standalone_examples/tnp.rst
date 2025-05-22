Tag and Probe
=============

The Tag and Probe is one of the tests that can be performed within the $H \rightarrow \gamma \gamma$ analysis. The processor that implements the main operations performed can be found at ``bottom_line/workflows/dystudies:TagAndProbeProcessor``.

To try it out, one can fetch the custom sample at ``/afs/cern.ch/user/g/gallim/public/HiggsDNA/HggNanoDY_10_6_26-das.json`` and run

.. code-block:: bash

   run_analysis.py --wf tagandprobe --dump path/to/directory --meta Era2017_legacy_xgb_v1 --samples HggNanoDY_10_6_26-das.json


**This example has to be rewritten since we need a sample JSON nowadays.**