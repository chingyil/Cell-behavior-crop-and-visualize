import numpy as np
import matplotlib.pyplot as plt

def f(z, l1=4, l2=3, l3=2, l4=3):
    assert len(z.shape) == 1
    z_len = z.shape[0]
    z_array = np.empty((z_len-l1+1,l1))
    for i in range(l1):
        z_array[:,i] = z[i:i+z_len-l1+1]
    y = np.median(z_array, axis=1)

    y_len = y.shape[0]
    y_array = np.empty((y_len-l2+1,l2))
    for i in range(l2):
        y_array[:,i] = y[i:i+y_len-l2+1]
    x = np.median(y_array, axis=1)

    x_len = x.shape[0]
    x_array = np.empty((x_len-l3+1,l3))
    for i in range(l3):
        x_array[:,i] = x[i:i+x_len-l3+1]
    v = np.median(x_array, axis=1)
    
    v_len = v.shape[0]
    v_array = np.empty((v_len-l4+1,l4))
    for i in range(l4):
        v_array[:,i] = v[i:i+v_len-l4+1]
    f = np.median(v_array, axis=1)
    
    return f

def median_filter(signal, l1=4, l2=3, l3=2, l4=3):
    x = np.empty_like(signal)
    x[:] = signal[:]
    l = x.shape[0]

    ll = l1 + l2 + l3 + l4 - 3
    crop = (ll - 1) // 2
    delta_signal = np.random.rand(l)
    f_signal = f(x, l1, l2, l3, l4)
    delta_signal = np.zeros_like(x)
    delta_signal[crop:l-crop] = x[ll//2:l-ll//2] - f_signal
    f_delta = f(delta_signal, l1, l2, l3, l4)
    u = np.empty_like(x)
    u[:] = x[:]
    u[crop:-crop] = f_signal + f_delta
    return u

if __name__ == '__main__':
    l = 30
    signal = np.random.rand(l)
    plt.plot(np.arange(l), signal, label='z')
    plt.plot(np.arange(l), median_filter(signal), label='u')
    plt.legend()
    plt.show()
