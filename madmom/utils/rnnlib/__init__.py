from rnnlib import create_nc_file, test_nc_files, RnnConfig

import os
import glob
NN_PATH = '%s/configs' % (os.path.dirname(__file__))

# generate lists of NN files
NN_ONSET_FILES = glob.glob("%s/onsets*save" % NN_PATH)
NN_BEAT_FILES = glob.glob("%s/beats*save" % NN_PATH)
