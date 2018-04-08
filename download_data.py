from __future__ import print_function

import os
import sys

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

URLBASE = 'https://storage.ramp.studio/astrophd_tutorial/{}'
DATA = [
    'data_train.npy',
    'data_test.npy',
    'labels_train.npy',
    'labels_test.npy'
]
MINIDATA = [
    'data_train_mini.npy',
    'data_test_mini.npy',
    'labels_train_mini.npy',
    'labels_test_mini.npy'
]


def main(output_dir='data'):
    filenames = MINIDATA

    try:
        if sys.argv[1].lower() in ['full', 'all']:
            filenames += DATA
    except IndexError:
        pass

    urls = [URLBASE.format(filename) for filename in filenames]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print("{} already downloaded".format(filename))
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()
