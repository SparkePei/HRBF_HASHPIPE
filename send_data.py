#!/usr/local/python

import struct,sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import struct
import socket

N_POL=2
N_STOKES=4
N_BEAM=int(sys.argv[1]) ## Number of beams we are forming
N_BEAM_STOKES=int(N_BEAM/N_POL) ## Number of Stokes beams we are forming
N_ELEM = int(sys.argv[2]) ## Number of elements/antennas in the array
N_TSAMP = int(sys.argv[3]) ## Number of time samples
N_FBIN = int(sys.argv[4])   ## Number of frequency bins
N_TSAMP_ACCU = int(sys.argv[5])  ## Number of decimated time samples per integrated beamformer output
N_ACCU=int(N_TSAMP/N_TSAMP_ACCU) ## Number of short time integrations
N_INPUTS=N_ELEM*N_FBIN*N_TSAMP ## Number of complex samples to process
N_WEIGHTS=N_ELEM*N_FBIN*N_BEAM ## Number of complex beamformer weights

#DIR = "/home/peix/cuda-workspace/test_cublasCgemmBatched/"
#DIR = "/home/peix/workspace/paf_sim/"
DIR = "/buff0/"
N_sample = 1024  # number of samples
N_element=7
#####################################################################
#################     read faked signal from file     ###############
sig=np.load('raw_data.npy') # Read faked signal from file
sig[4]=sig[7]
sig=sig[0:N_element]
F_sig=fft.rfft(sig) # trun to frequency complex
F_sig=F_sig[:,1:]

#####################################################################

PKTSIZE = 4096
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1",5009)
fp=open('file0000','r')
data=numpy.fromstring(fp.read(),dtype='b')
for i in range(0,len(data)/PKTSIZE):
	#sock.sendto(str(data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
	n=sock.sendto((data[i*PKTSIZE:(i+1)*PKTSIZE]),addr)
	print "send %d bytes of number %d packets to local address! "%(n,i)
	time.sleep(0.01) #0.000001 sec(100ns) no packets loss
fp.close()
#####################################################################
###     write data to file, transpose data needed in CUDA code    ###
f_data = open(DIR+"data_"+str(N_FBIN)+"x"+str(N_ELEM)+"x"+str(N_TSAMP)+".bin","wb")

data_tmp = N_TSAMP*[N_ELEM*[2*N_FBIN*[0]]]
data_tmp = np.array(data_tmp)
for i in range(N_TSAMP):
    for j in range(N_ELEM):
        #data_tmp=2*N_FBIN*[0]
        if(j<14):
            for k in range(N_FBIN):
                #data_tmp_complex[i][j][k] = F_sig[int(j/2)][94+k]
                data_tmp[i][j][2*k] = F_sig[int(j/2)][94+k].real
                data_tmp[i][j][2*k+1] = F_sig[int(j/2)][94+k].imag
            #print(data_tmp[i][j])
data_frame=np.reshape(data_tmp,N_TSAMP*N_ELEM*N_FBIN*2)
print(np.shape(data_frame))
data_frame_pack = struct.pack(str( N_TSAMP*N_ELEM*N_FBIN*2)+'f',*data_frame)
f_data.write(data_frame_pack)
print("Write data to file: done!")
f_data.close()
