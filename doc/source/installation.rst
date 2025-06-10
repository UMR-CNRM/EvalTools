Installation
============

Get source code
---------------

`<https://github.com/UMR-CNRM/EvalTools>`_

Install with pip
----------------

Go to the decompressed package directory and execute

.. code-block:: sh

    pip3 install .

In this case, some python packages need to be pre-installed:

* cython==3.0.12
* numpy==2.2.4
* scipy==1.15.2
* matplotlib==3.10.1
* pandas==2.2.3
* netCDF4==1.7.2
* cartopy==0.24.1

Given version numbers are those used to develop and test this `evaltools` release, but feel free to try with other versions.

If you want to install the dependencies, you can try

.. code-block:: sh

    pip3 install -r requirements.txt
    pip3 install -r optional-requirements.txt

.. important:: In this case, it is recommended to install **evaltools** inside a Python virtual environment so that dependencies will be re-installed with required versions without any conflict.
