Instructions for Developers
===========================

----------
Contribute
----------

Please lint, format and run tests before sending a PR:

.. code-block:: bash

   flake8 bottom_line
   black bottom_line
   pytest

We follow certain conventions in the codebase. Please make sure to follow them when contributing:

#. Make use of the common abbreviations for frequent packages, e.g., ``np`` for ``numpy`` and ``ak`` for ``awkward``.
#. When using the ``argparse`` package, use kebab case for the argument names, e.g., ``--input-file``. Do not use snake case (underscores). Note: Internally, this is converted back to snake case, so you access the argument with ``args.input_file``, but using kebab case in argparse is the unix convention.

--------------------
Update Documentation
--------------------

When the package is modified, it is highly recommended to also update the documentation accordingly. The files that make the docs that you are reading are located inside ``docs/source``. You can modify them or add new ones.

When you are satisfied with the result, do the following:

.. code-block:: bash

   cd docs
   sphinx-apidoc -o source/modules ../bottom_line
   sphinx-build source build/html

The ``sphinx-apidoc`` command will build the documentation from the package's docstrings, so every change in the package itself will be picked up.
At this point you can see (locally) how the updated docs look like by simply opening the just built html section using your favourite browser, e.g.:

.. code-block:: bash

   firefox build/html/index.html

---------------------------------
Debug Unexpected Awkward Behavior
---------------------------------

It is not always easy to understand from within the processor itself what is wrong with the new processor that we just wrote. In these cases, you can try to reproduce interactively what happens inside the ``process`` function of the ``bbMETBaseProcessor`` with the following lines.

.. code-block:: python

   >>> from coffea import nanoevents
   >>> f_name = "path_to_nanoaod_file.root"
   >>> events = nanoevents.NanoEventsFactory.from_root(f_name).events()
   >>> photons = events.Photon


``events`` and ``photons`` are the main variables that you need to use and from here you can start to debug the processor.