import logging
import gzip
import struct
import os
import os.path as path
import cv2

import numpy as np
import cxflow as cx

DOWNLOAD_ROOT = 'https://github.com/Cognexa/cxflow-examples/releases/download/mnist-dataset/'
FILENAMES = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}

class MNISTDataset(cx.DownloadableDataset):
    """ MNIST dataset for hand-written digits recognition."""
    
    def _configure_dataset(self, data_root=path.join('mnist_convnet', '.mnist-data'), batch_size:int=50, **kwargs) -> None:
        self._batch_size = batch_size
        self._data_root = data_root
        self._download_urls = [path.join(DOWNLOAD_ROOT, filename) for filename in FILENAMES.values()]
        self._data = {}
        self._data_loaded = False

    def _load_data(self) -> None:
        if not self._data_loaded:
            logging.info('Loading MNIST data to memory')
            for key in FILENAMES:
                file_path = path.join(self._data_root, FILENAMES[key])
                if not path.exists(file_path):
                    raise FileNotFoundError('File `{}` does not exist. '
                                            'Run `cxflow dataset download <path-to-config>` first!'.format(file_path))
                with gzip.open(file_path, 'rb') as file:
                    if 'images' in key:
                        _, _, rows, cols = struct.unpack(">IIII", file.read(16))
                        self._data[key] = np.frombuffer(file.read(), dtype=np.uint8).reshape(-1, rows, cols, 1)
                    else:
                        _ = struct.unpack(">II", file.read(8))
                        self._data[key] = np.frombuffer(file.read(), dtype=np.int8)
            self._data_loaded = True

    def train_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['train_labels']), self._batch_size):
            yield {'images': self._data['train_images'][i: i + self._batch_size],
                   'labels': self._data['train_labels'][i: i + self._batch_size]}

    def test_stream(self) -> cx.Stream:
        self._load_data()
        for i in range(0, len(self._data['test_labels']), self._batch_size):
            yield {'images': self._data['test_images'][i: i + self._batch_size],
                   'labels': self._data['test_labels'][i: i + self._batch_size]}
    def predict_stream(self) -> cx.Stream:
    
        """ NO PROBLEM WHEN READING ALL DATA INTO MEMORY AND THEN PREDICT """
        """
        all_data = np.zeros((1000, 28, 28, 1))
        for i in range(0, 1000):
            all_data[i, :, :, :] = np.load('./dataMnist/' + str(i) + '.npy')   
        idx = 0
        bid = 0
        for i in range(0, 20):
            cv2.imwrite('./output/' + str(bid) + 'a.jpeg',all_data[i*50, :, :, :])
            yield {'images': all_data[i*50:(i+1)*50, :, :, :], 'id': str(bid)}    
            bid += 1
        """


        """ WHEN READING ON DEMAND THE FIRST FEW ARE NOT RIGHT """

        idx = 0
        bid = 0
        images = np.zeros((self._batch_size, 28, 28, 1))
        frame = 0
        while idx < 1000:
            images[frame, :, :] = np.load('./dataMnist/' + str(idx) + '.npy')
            frame += 1
            
            if frame == self._batch_size:
                frame =  0
                cv2.imwrite('./output/' + str(bid) + 'a.jpeg', images[0, :, :, :])
                yield {'images': images, 'id': str(bid)}
                bid += 1

            idx += 1

