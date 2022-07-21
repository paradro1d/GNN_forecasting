#!/bin/bash

#Preprocesses grib file into suitable tfrecords format

#Input format:
#bash convert_to_tfrecords.sh <grib filename> <longitude resolution> <latitude resolution> <longs filename> <lats filename> <length of timeseries in grib>

COMMAND=$(cat <<EOF
from grib_to_tfrecords import grib_to_tfrecords
import numpy as np
lats = np.load("$5")
longs = np.load("$4")

grib_to_tfrecords("$1", $2, $3, longs, lats, $6)
EOF
)
python3 -c "$COMMAND"
rm part_*.tfrecords
