Development
===========

As an open-source project by researchers for researchers, we highly welcome
any contribution!

What to contribute
------------------

Give feedback
~~~~~~~~~~~~~

To send us general feedback, questions or ideas for improvement, please post on
`our mailing list`_.

Report bugs
~~~~~~~~~~~

Please report any bugs at the `issue tracker on GitHub`_.
If you are reporting a bug, please include:

* your version of madmom,
* steps to reproduce the bug, ideally reduced to as few commands as possible,
* the results you obtain, and the results you expected instead.

If you are unsure whether the experienced behaviour is intended or a bug,
please just ask on `our mailing list`_ first.

Fix bugs
~~~~~~~~

Look for anything tagged with "bug" on the `issue tracker on GitHub`_ and fix
it.

Features
~~~~~~~~

Please do not hesitate to propose any ideas at the `issue tracker on GitHub`_.
Think about posting them on `our mailing list`_ first, so we can discuss it
and/or guide you through the implementation.

Alternatively, you can look for anything tagged with "feature request" or
"enhancement" on the `issue tracker on GitHub`_.

.. _write_documentation:

Write documentation
~~~~~~~~~~~~~~~~~~~

Whenever you find something not explained well, misleading or just wrong,
please update it! The *Edit on GitHub* link on the top right of every
documentation page and the *[source]* link for every documented entity
in the API reference will help you to quickly locate the origin of any text.

How to contribute
-----------------

Edit on GitHub
~~~~~~~~~~~~~~

As a very easy way of just fixing issues in the documentation, use the *Edit
on GitHub* link on the top right of a documentation page or the *[source]* link
of an entity in the API reference to open the corresponding source file in
GitHub, then click the *Edit this file* link to edit the file in your browser
and send us a Pull Request.

For any more substantial changes, please follow the steps below.

Fork the project
~~~~~~~~~~~~~~~~

First, fork the project on `GitHub`_.

Then, follow the :doc:`general installation instructions <installation>` and,
more specifically, the :ref:`installation from source <install_from_source>`.
Please note that you should clone from your fork instead.

Documentation
~~~~~~~~~~~~~

The documentation is generated with `Sphinx
<http://sphinx-doc.org/latest/index.html>`_. To build it locally, run the
following commands::

    cd docs
    make html

Afterwards, open ``docs/_build/html/index.html`` to view the documentation as
it would appear on `readthedocs <http://madmom.readthedocs.org/>`_. If you
changed a lot and seem to get misleading error messages or warnings, run
``make clean html`` to force Sphinx to recreate all files from scratch.

When writing docstrings, follow existing documentation as much as possible to
ensure consistency throughout the library. For additional information on the
syntax and conventions used, please refer to the following documents:

* `reStructuredText Primer <http://sphinx-doc.org/rest.html>`_
* `Sphinx reST markup constructs <http://sphinx-doc.org/markup/index.html>`_
* `A Guide to NumPy/SciPy Documentation
  <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_


.. _GitHub: https://github.com/CPJKU/madmom
.. _issue tracker on GitHub: https://github.com/CPJKU/madmom/issues
.. _our mailing list: https://groups.google.com/d/forum/madmom-users
