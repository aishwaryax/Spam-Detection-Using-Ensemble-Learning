from __future__ import print_function
import os, sys
import zipfile

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

DATA_DIR = '.\Glove'

def download_and_extract():
    DATA_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    path = os.path.abspath(dest_directory)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath)


        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(DATA_DIR)
        zip_ref.close()
    return path