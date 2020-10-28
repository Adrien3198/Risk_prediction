.. Risk prediction documentation master file, created by
   sphinx-quickstart on Mon Oct 26 16:16:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Risk prediction's documentation!
*******************************************

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


evaluation.py
=============

.. automodule:: evaluation
   :members:


data_preparation.py
===========================

.. automodule:: data_preparation
   :members:


random_forest_classifier.py
===========================

A python script to build a random forest classifier model
::
 $ python random_forest_classifier.py <max_depth> <min_sample_split>

.. automodule:: random_forest_classifier
   :members:


gradient_boosting_classifier.py
===============================
A python script to build a random forest classifier model
::
 $ python gradient_boosting_classifier.py <alpha> <max_depth> <min_samples_split>

.. automodule:: gradient_boosting_classifier
   :members:


xgboost_classifier.py
=====================
A python script to build a xgboost classifier model
:: 
 $ python xgboost_classifier.py <alpha> <max_depth> <min_child_weigth>

.. automodule:: xgboost_classifier
   :members: