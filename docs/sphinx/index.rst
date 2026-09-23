..
    # Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
    # contributors. See the CHAI LICENSE and COPYRIGHT files for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

******
CHAI
******

CHAI is a C++ libary providing an array object that can be used transparently
in multiple memory spaces. Data is automatically migrated based on
copy-construction, allowing for correct data access regardless of location.
CHAI can be used standalone, but is best when paired with the RAJA library,
which has built-in CHAI integration that takes care of everything.

- If you want to get and install CHAI, take a look at our getting started
  guide. 
- If you are looking for documentation about a particular CHAI function, see
  the code documentation.
- Want to contribute? Take a look at our developer and contribution guides.

Any questions? Contact chai-dev@llnl.gov

.. toctree::
  :maxdepth: 2
  :caption: Basics

  getting_started
  tutorial
  user_guide
  execution_contexts

.. toctree::
  :maxdepth: 2
  :caption: Reference

  advanced_configuration
  code_documentation

.. toctree::
  :maxdepth: 2
  :caption: Contributing

  contribution_guide
  developer_guide
