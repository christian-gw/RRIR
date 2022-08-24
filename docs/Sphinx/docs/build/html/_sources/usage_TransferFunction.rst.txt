Example use cases of the TransferFunction Class
===============================================


Creation of Transfer function of two signals
--------------------------------------------
You have two signals of which you want to calculate the transferfunction/quotient.

.. code-block:: Python
   :linenos:

   # Import
   from RIRR import Signal
   from RIRR import TransferFunction

   # Load incomming and outgoing signal
   incomming = Signal(...)
   outgoing = Signal(...)

   # Generate TransferFuction
   tf = TransferFunction(incoming_signal=incomming,
                         reflected_signal=outgoing)
   
   # Generate new Transferfunction on Terz (1/3 Octave) base
   tf_oct = tf.get_okt(fact=1/3)

   # Plot both
   fig, ax = tf.plot_hf()
   fig, ax = tf_oct.plot_hf()

