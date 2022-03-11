import numpy as np
import h5py


def write_hdf5(file, tensor, key = 'tensor'):
    """Write a simple tensor, i.e. numpy array ,to HDF5.
    """
    assert type(tensor) == np.ndarray, 'expects numpy.ndarray'

    h5f = h5py.File(file, 'w')
    chunks = list(tensor.shape)
    if len(chunks) > 2:
        chunks[2] = 1
        if len(chunks) > 3:
            chunks[3] = 1
            if len(chunks) > 4:
                chunks[4] = 1

    h5f.create_dataset(
        key, data = tensor, chunks = tuple(chunks), compression = 'gzip'
    )
    h5f.close()


def read_hdf5(file, key = 'tensor'):
    """Read a tensor, i.e. numpy array, from HDF5.
    """
    assert os.path.exists(file), 'file %s not found' % file
    h5f = h5py.File(file, 'r')
    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    tensor = h5f[key][()]
    h5f.close()
    return tensor
