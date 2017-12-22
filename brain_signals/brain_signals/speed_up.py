import numpy as np
from scipy.signal.sigtools import _linear_filter

def filtfilt(b, a, x, zi):
    axis = -1
    max_a_b = max(len(a), len(b))
    # edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=max(len(a), len(b)))
    # unfloded below
    edge = max_a_b * 3
    # left_end = axis_slice(x, start=0, stop=1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(0, 1, None)
    left_end = x[a_slice]

    # left_ext = axis_slice(x, start=edge, stop=0, step=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(edge, 0, -1)
    left_ext = x[a_slice]

    # right_end = axis_slice(x, start=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(-1, None, None)
    right_end = x[a_slice]

    # right_ext = axis_slice(x, start=-2, stop=-(edge + 2), step=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(-2, -(edge +2), -1)
    right_ext = x[a_slice]

    ext = np.concatenate((2 * left_end - left_ext, x, 2 * right_end - right_ext), axis=axis)
  
    # NEW UNFOLDED SECTION
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    # x0 = axis_slice(ext, stop=1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * ext.ndim
    a_slice[axis] = slice(None, 1, None)
    x0 = ext[a_slice]

    # NEW UNFOLDED SECTION
    # (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)
    # unfolded with
    (y, zf) = _linear_filter(b, a, ext, axis, zi * x0)

    # NEW UNFOLDED SECTION
    # y0 = axis_slice(y, start=-1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(-1, None, None)
    y0 = y[a_slice]

    # NEW UNFOLDED SECTION
    # y1 = axis_reverse(y, axis=axis)
    # unfolded with
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(None, None, -1)
    y1 = y[a_slice]

    # NEW UNFOLDED SECTION
    # (y, zf) = lfilter(b, a, y1, axis=axis, zi=zi * y0)
    # unfolded below
    (y, zf) = _linear_filter(b, a, y1, axis, zi * y0)

    # NEW UNFOLDED SECTION
    # y = axis_reverse(y, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(None, None, -1)
    y = y[a_slice]

    if edge > 0:
        # NEW UNFOLDED SECTION
        # y = axis_slice(y, start=edge, stop=-edge, axis=axis)
        # unfolded below
        a_slice = [slice(None)] * y.ndim
        a_slice[axis] = slice(edge, -edge, None)
        y = y[a_slice]

    return y

'''
import numpy as np
from scipy.signal.signaltools import _validate_pad, lfilter_zi, lfilter
from scipy.signal._arraytools import axis_reverse, axis_slice
from scipy.signal.sigtools import _linear_filter


# axis_slice:
# a_slice = [slice(None)] * {name_of_caller_arg}.ndim
# a_slice[axis] = slice(None, None, None)
# {name_of_result} = {name_of_caller_arg}[a_slice]

def filtfilt_unfolded(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
    # edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=max(len(a), len(b)))
    # unfloded below
    max_a_b = max(len(a), len(b))
    edge = max_a_b * 3
    # left_end = axis_slice(x, start=0, stop=1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(0, 1, None)
    left_end = x[a_slice]

    # left_ext = axis_slice(x, start=edge, stop=0, step=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(edge, 0, -1)
    left_ext = x[a_slice]

    # right_end = axis_slice(x, start=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(-1, None, None)
    right_end = x[a_slice]

    # right_ext = axis_slice(x, start=-2, stop=-(edge + 2), step=-1, axis=axis)
    # unfolded below
    #-> ignore a_slice = [slice(None)] * x.ndim
    a_slice[axis] = slice(-2, -(edge +2), -1)
    right_ext = x[a_slice]

    ext = np.concatenate((2 * left_end - left_ext, x, 2 * right_end - right_ext), axis=axis)
  
    # NEW UNFOLDED SECTION
    # This section can be computed only once for each band, and passed along as a parameter
    # zi = lfilter_zi(b, a)
    # unfolded below

    # linalg.companion(a)
    # unfolded below
    first_row = -a[1:] / (1.0 * a[0])
    n = a.size
    c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    c[list(range(1, n - 1)), list(range(0, n - 2))] = 1

    IminusA = np.eye(max_a_b - 1) - c.T
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = np.linalg.solve(IminusA, B)

    # NEW UNFOLDED SECTION
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    # x0 = axis_slice(ext, stop=1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * ext.ndim
    a_slice[axis] = slice(None, 1, None)
    x0 = ext[a_slice]

    # NEW UNFOLDED SECTION
    # (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)
    # unfolded with
    (y, zf) = _linear_filter(b, a, ext, axis, zi * x0)

    # NEW UNFOLDED SECTION
    # y0 = axis_slice(y, start=-1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(-1, None, None)
    y0 = y[a_slice]

    # NEW UNFOLDED SECTION
    # y1 = axis_reverse(y, axis=axis)
    # unfolded with
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(None, None, -1)
    y1 = y[a_slice]

    # NEW UNFOLDED SECTION
    # (y, zf) = lfilter(b, a, y1, axis=axis, zi=zi * y0)
    # unfolded below
    (y, zf) = _linear_filter(b, a, y1, axis, zi * y0)

    # NEW UNFOLDED SECTION
    # y = axis_reverse(y, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[axis] = slice(None, None, -1)
    y = y[a_slice]

    if edge > 0:
        # NEW UNFOLDED SECTION
        # y = axis_slice(y, start=edge, stop=-edge, axis=axis)
        # unfolded below
        a_slice = [slice(None)] * y.ndim
        a_slice[axis] = slice(edge, -edge, None)
        y = y[a_slice]

    return y

def filtfilt_original(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):

    # method == "pad"
    edge, ext = _validate_pad(padtype, padlen, x, axis, ntaps=max(len(a), len(b)))

    # Get the steady state of the filter's step response.
    zi = lfilter_zi(b, a)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    x0 = axis_slice(ext, stop=1, axis=axis)

    # Forward filter.
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    y0 = axis_slice(y, start=-1, axis=axis)
    (y, zf) = lfilter(b, a, axis_reverse(y, axis=axis), axis=axis, zi=zi * y0)

    # Reverse y.
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return y

def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None):
#    filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, irlen=None)

    # method == "pad"
    # edge, ext = _validate_pad('odd', None, x, -1, ntaps=max(len(a), len(b)))
    # unfolded below

    max_a_b = max(len(a), len(b))

    edge = max_a_b * 3

    a_slice = [slice(None)] * x.ndim
    a_slice[-1] = slice(0, 1, None)
    left_end = x[a_slice]
    a_slice[-1] = slice(edge, 0, -1)
    left_ext = x[a_slice]
    a_slice[-1] = slice(-1, None, None)
    right_end = x[a_slice]
    a_slice[-1] = slice(-2, -(edge + 2), -1)
    right_ext = x[a_slice]

    ext = np.concatenate((2 * left_end - left_ext,
                          x,
                          2 * right_end - right_ext),
                         axis=-1)

    # Get the steady state of the filter's step response.
    # zi = lfilter_zi(b, a)
    # unfolded below

    # linalg.companion(a)
    # unfolded below
    first_row = -a[1:] / (1.0 * a[0])
    n = a.size
    c = np.zeros((n - 1, n - 1), dtype=first_row.dtype)
    c[0] = first_row
    c[list(range(1, n - 1)), list(range(0, n - 2))] = 1

    IminusA = np.eye(max_a_b - 1) - c.T
    B = b[1:] - a[1:] * b[0]
    # Solve zi = A*zi + B
    zi = np.linalg.solve(IminusA, B)

    # Reshape zi and create x0 so that zi*x0 broadcasts
    # to the correct value for the 'zi' keyword argument
    # to lfilter.
    zi_shape = [1] * x.ndim
    zi_shape[axis] = zi.size
    zi = np.reshape(zi, zi_shape)
    # x0 = axis_slice(ext, stop=1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * ext.ndim
    a_slice[-1] = slice(None, 1, None)
    x0 = ext[a_slice]

    # Forward filter.
    (y, zf) = lfilter(b, a, ext, axis=axis, zi=zi * x0)

    # Backward filter.
    # Create y0 so zi*y0 broadcasts appropriately.
    # y0 = axis_slice(y, start=-1, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[-1] = slice(-1, None, None)
    y0 = y[a_slice]

    # y1 = axis_reverse(y, axis=axis)
    # unfolded below
    a_slice = [slice(None)] * y.ndim
    a_slice[-1] = slice(None, None, -1)
    y1 = y[a_slice]

    (y, zf) = lfilter(b, a, y1, axis=axis, zi=zi * y0)

    # Reverse y.
    # y = axis_reverse(y, axis=axis)
    a_slice = [slice(None)] * y.ndim
    a_slice[-1] = slice(None, None, -1)
    y = y[a_slice]

    if edge > 0:
        # Slice the actual signal from the extended signal.
        # y = axis_slice(y, start=edge, stop=-edge, axis=axis)
        # unfolded below
        a_slice = [slice(None)] * y.ndim
        a_slice[-1] = slice(edge, edge, None)
        y = y[a_slice]

    return y
'''