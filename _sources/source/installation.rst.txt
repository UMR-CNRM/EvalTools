Installation
============

Get source code
---------------

`<https://github.com/UMR-CNRM/EvalTools>`_

Install with pip
----------------

Several python libraries are required to run evaltools. To install them,
you can do

.. code-block:: sh

    pip3 install -r requirements.txt


Optionally, you can also install

.. code-block:: sh

    pip3 install -r optional-requirements.txt

which are dependencies needed for plotting maps or build the documentation.

The version numbers specified inside these two requirement files are the ones
that have been used to test the library.

Then, go to the package directory and execute

.. code-block:: sh

    pip3 install . --no-build-isolation
