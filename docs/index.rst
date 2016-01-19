Welcome to madmom!
==================

Madmom is an audio signal processing library written in Python with a strong
focus on music information retrieval (MIR) tasks. The project is on `GitHub`_.

It's main features / design goals are:

* ease of use,
* rapid prototyping of signal processing workflows,
* most things are modeled as numpy arrays (enhanced by additional methods and
  attributes),
* simple conversion to a running program by the use of processors.

Madmom is a work in progress, input is always welcome. The available
documentation is limited for now, but :ref:`you can help to improve it
<write_documentation>`.

The documentation is split into several parts:

* The :ref:`user_guide` explains how to install, use and conrtibute to madmom.
* If you are looking for information on a specific function, class or method,
  the :ref:`api_reference` is for you.

.. _user_guide:

.. toctree::
  :maxdepth: 2
  :caption: User Guide

  installation
  development

.. _api_reference:

.. toctree::
  :maxdepth: 3
  :caption: API Reference

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

Acknowledgements
----------------

Supported by the European Commission through the `GiantSteps project
<http://www.giantsteps-project.eu>`_ (FP7 grant agreement no. 610591) and the
`Phenicx project <http://phenicx.upf.edu>`_ (FP7 grant agreement no. 601166)
as well as the `Austrian Science Fund (FWF) <https://www.fwf.ac.at>`_ project
Z159.

.. _GitHub: https://github.com/CPJKU/madmom
