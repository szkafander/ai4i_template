.. lkab_slag_ai documentation master file, created by
   sphinx-quickstart on Wed May  8 05:57:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lkab_slag_ai's documentation!
========================================
Quick start guide:

#. Copy image data from LKCAMADM01 to the /data/image/excerpt... folder using the copynth_recursive Powershell script
#. Copy process data CSV's to the /data/process/KK4 folder
#. Go to the scripts folder and start running scripts

Suggested order of running scripts:

#. ``daily_to_cont`` in the image folder unless the image set is already continuous (i.e., not in separate daily folders).
#. ``db_script`` to create and populate the process database from the CSV's.
#. ``generate_in_memory_process_data_array`` to save the process data array.
#. ``train_discriminator_network`` to train the discriminator network. This network will help in producing the temporally averaged images.
#. ``filter_images`` to get the temporally averaged images.
#. here you can create or add your own training images.
#. ``train_segmenter_network`` to train the segmenter network.
#. ``extract_slag_signal`` to get the slag signal. You may need to rerun db_script to put this into the database (optional).
#. ``train_process_model`` to train the process model.
#. ``sensitivity_analysis`` to visualize results from the process model.

Otherwise, explore the documentation of the modules below.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
