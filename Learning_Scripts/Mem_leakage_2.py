# %%
import numpy as np
import gc


@profile
def profile_fun():

    class Sig_Dummy_Class:
        def __init__(self, number):
            self.inner_np_array = np.linspace(0, 10, number)
            self.second_array = 2*self.inner_np_array
            self.len = number

        def __del__(self):
            print('Destructor is called.')
            del self.inner_np_array, self.len
            gc.collect()

    Sig_Dummy = Sig_Dummy_Class(1000000)
    del Sig_Dummy


if __name__ == '__main__':
    profile_fun()

# %%
