#!/usr/bin/env python

import glob
import os
import opendeplete

decay_files = sorted(glob.glob('/opt/data/endf/endf-b-vii.1/decay/*.endf'))
nfy_files = sorted(glob.glob('/opt/data/endf/endf-b-vii.1/nfy/*.endf'))
neutron_files = sorted(glob.glob('/opt/data/endf/endf-b-vii.1/neutrons/*.endf'))

chain = opendeplete.DepletionChain.from_endf(decay_files, nfy_files, neutron_files)
chain.xml_write('chain_new.xml')
