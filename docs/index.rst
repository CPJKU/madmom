Welcome to madmom!
==================

Madmom is an audio signal processing library written in Python with a strong
focus on music information retrieval (MIR) tasks. The project is on `GitHub`_.

It's main features / design goals are:

* ease of use,
* rapid prototyping of signal processing workflows,
* most stuff is modeled as numpy arrays (enhanced by additional methods and
  attributes),
* simple conversion to a running program by the use of processors.

Madmom is a work in progress, input is always welcome. The available
documentation is limited for now, but
:doc:`you can help to improve it </development>`.

User Guide
----------

The madmom user guide explains how to install madmom, how to get things done
and how to contribute to the library as a developer.

.. toctree::
  :maxdepth: 1

  installation
  tutorial
  development


API Reference
-------------

If you are looking for information on a specific function, class or method,
this part of the documentation is for you.

.. toctree::
  :maxdepth: 2

  modules/audio
  modules/features
  modules/evaluation
  modules/ml
  modules/utils
  modules/processors

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/CPJKU/madmom
