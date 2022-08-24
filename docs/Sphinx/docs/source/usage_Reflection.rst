Example use cases of the Measurement and MeasurementPoint Classes in the Reflection module
==========================================================================================

Both of the classes of the reflection module serve the purpose of analysing Reflection measurements.
Basis is the norm DIN EN 1793.

Definition of Measurement with multiple points + Correction + Averaging
-----------------------------------------------------------------------

You have two signals loaded from wav files representing **impulses** of a reflection measurement.
You also have one Signal aquired with lots of space around.
So the earliest reflections arive much later than in the former two measurements.

.. code-block:: Python
   :linenos:

   # Import
   from RIRR import Signal
   from RIRR import TransferFunction
   from RIRR import Measurement, MeasurementPoint

   # Define inforamtion for measurements
   #      Key        [path,  in_win, re_win]
   NR = {'Wand_0_0': ['...', 0.1234, 0.1245],
         'Wand_+_0': ['...', 0.1234, 0.1245]}
   #       Key          x,   y
   POS = {'Wand_0_0':  [0,   0],
          'Wand_+_0':  [.4,   0]}

   # Load the Data with the dict keys as filenames
   NAME = 'IMP_%s.wav'

   # Set up Mesurement object with distances Source --- Mic --- Wall
   mea_Marth = Measurement('Measurement_Martha_Kirche', 1., .25)

   for i, position in zip(range(len(NR)), NR.keys()):
      # Load impulse Files
      NR[position].append(Signal(path=AVG_DIR,
                                 name=NAME %(position)))

      # Create TF using Adrienne windowing
      arr = NR[position]
      NR[position].append(TransferFuction(signal=arr[-1],
                                          in_win=arr[1],
                                          re_win=arr[2]))
   
      # Add Measurement Points to Measurement
      mea_Marth.create_mp(i,
                          NR[position][3].get_octave_band(fact=1),
                          POS[position])
      
      mea_Marth.mp_lst[i].apply_c()

   # Average and plot
   mea_Marth.average_mp()
   fig, ax = mea_Marth.average.plot_hf()