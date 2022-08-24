Example use cases of the Signal Class
=====================================

Installation
------------

To use RIRR, first install it and its dependencies using pip or conda:

.. code-block:: console

   $ pip liborsa # Not in conda
   $ pip install ./path/to/RIRR/folder
   $ # Alternative
   $ conda install ./path/to/RIRR/folder

Creation of Impulse Responses from Sweeps
-----------------------------------------
You have measured a Sweep response in a room using a exponential sweep.
The e-sweep starts at start_freq, ends at end_freq and lasts sweep_duration.
You saved the sweep at ".path/to/wav" and called it "filename.wav".

After you executed the following code, there is a new file "filename.wav" in your files folder.
This file contains the transformed impulse response.

.. code-block:: Python
   :linenos:

   # Import
   from RIRR import Signal

   # Load Measured Data
   mea_signal = Signal(path="./path/to/wav",
                       name="filename.wav")

   # Specify Exitation Sweep Signal
   ex_signal = Signal(par_sweep=(start_freq,
                      sweep_duration,
                      end_freq))

   # Deconvolve Exitation to get Impulse
   impulse = mea_signal.impulse_response(ex_signal)
   
   # Save Impulse to current working directory
   impulse.write_wav("filename.wav",
                     F_samp=48e3,
                     norm=True)

Plot Impulse Response
---------------------

After you performed the last step and loaded and transformed the file (impulse.write_wav is optional),
you can visualise the impulse.

.. code-block:: Python
   :linenos:

   # Time plot, frequency plot and spectrogram
   # Returns figure and axis object to further work with
   impulse.plot_y_t()
   impulse.plot_y_f()
   impulse.plot_spec_transform()

Cut and/or average multiple impulses
------------------------------------

Very often you will encounter multiple impulse responses in one or more files.
This example deals with cutting up one wav into its single impulses and averages them.

Assuming you loaded a measurement and transformed it to a impulse like shown in 'Creation of Impulse Responses from Sweeps'
Further assuming you plotted it like in 'Plot Impulse Response' and learned where the individual impulses started.

.. code-block:: Python
   :linenos:

   # Cut the individual impulses
   # The times (e.g. 3, 10) are examples and should be changed
   all_cut = []
   all_cut.append(impulse.cut_signal(3, 10))
   all_cut.append(impulse.cut_signal(17, 24))
   all_cut.append(impulse.cut_signal(33, 40))

   # Upsampling and synchronisation of impulses
   # Upsampling increases temporal fit
   # Before Upsampling save the previous sampling rate
   in_Sample = all_cut[0].dt
   cut_up = [imp.resample(F_up) for imp in all_cut]
   rotate_sig_lst(cut_up)

   # Averaging of the impulses and write to new signal object
   imp_avg = Signal(signal_lst_imp=cut_up)

   # Perform downsamping
   imp_avg_down = imp_avg.resample(in_Sample)

Time based Key Values
---------------------

Calculation of the t20 and c50 values on the averaged impulse response from last section.

.. code-block:: Python
   :linenos:

   print('T20: ' + str(imp_avg_down.txx(xx=20)))
   print('C50: ' + str(imp_avg_down.cxx(xx=50)))
