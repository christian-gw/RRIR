# %% 
# Import using Sys for direct Import
import sys
sys.path.append("C:\\Users\\gmeinwieserch\\Documents\\Python\\Reflection\\sml\\")
import Signal
import ctypes
import gc
import numpy as np

@profile
def prof_fun():
    test_array = np.array([1.2, 2.3, 3.4, 4.5])
    sweep_dummy = Signal.Signal(par_sweep=(50, 10, 5000),
                                dt=1/48e3)

    del test_array
    del sweep_dummy.y

# %%

if __name__ == "__main__":
    prof_fun()
else:
    test_array = np.array([1.2, 2.3, 3.4, 4.5])
    sweep_dummy = Signal.Signal(par_sweep=(50, 10, 5000),
                            dt=1/48e3)

    # Interesting: Size of Arrays does not count for size of obj
    print("ID of Signal: %d" % id(sweep_dummy))
    print("Size of Signal: %d" % sys.getsizeof(sweep_dummy))

    print('ID of y: %d' % id(sweep_dummy.y))
    print('Size of y: %d' % sys.getsizeof(sweep_dummy.y))

    print('ID of y_f: %d' % id(sweep_dummy.y_f))
    print('Size of y_f: %d' % sys.getsizeof(sweep_dummy.y_f))

    print('ID of t: %d' % id(sweep_dummy.axis_arrays['t']))
    print('Size of t: %d' % sys.getsizeof(sweep_dummy.axis_arrays['t']))

    print('ID of xf: %d' % id(sweep_dummy.axis_arrays['xf']))
    print('Size of xf: %d' % sys.getsizeof(sweep_dummy.axis_arrays['xf']))

# %%
# a = id(sweep_dummy.y)
# print('References to sweep_dummy.y before del:')
# print(gc.get_referrers(ctypes.cast(a, ctypes.py_object).value))

# del sweep_dummy

# print('asdf')
# %% 
# del sweep_dummy
# gc.collect()

# print('\n\nReferences to sweep_dummy.y after del:')
# print(gc.get_referrers(ctypes.cast(a, ctypes.py_object).value))

# # print(ctypes.cast(a, ctypes.py_object).value)
# %%
