# hrbf_Pipeline_Hashpipe
### Introduction
    
* This  code is used to receive the packets from high speed ethernet, perform beamforming and Stokes calculation and save the accumulated data to disk.The packets are transmitted from RFSoC boards through 100 Gb Ethernet. And the output data will be stored in NVMe SSD cards to get a high data storage rate.<br>
* Three threads are developed to perform packet receiving, beamforming calculation and data storage, which are:
    * hrbf_net_thread; (packet receving)
    * hrbf_gpu_thread; (GPU beamforming calculation)
    * hrbf_output_thread; (save beamformed and accumulated data to disk)

* There are 2 buffers between three threads, each of them has a 3 segments ring buffer.  Buffer status could be abstracted from each ring buffer. There is a demo about how does Hashpipe working you can find [in here](https://github.com/SparkePei/demo1_hashpipe).

### Installation
* Required packages as follows:
    ```
    Hashpipe 1.5
    Ruby 2.1.10
    rb-hashpipe 1.5
    ```
    [Here](https://github.com/SparkePei/demo1_hashpipe) is the tutorial to install these packages.
* once these required packages installed properly, you can download this software from github:
    ```
    git clone https://github.com/SparkePei/hrbf_hashpipe.git
    ```
* enter this directory and run:
    ```
    make
    sudo make install
    ```
### How to run this software
* You can easily tap following command to start at your installation directory:
    ```
    ./hrbf_init.sh
    ```
    Inside this shell script, you can set as many as four instances to start simultaneously in a for loop:
    ```
    for i in 0 1 2 3 
        do
            hashpipe -p ./hrbf_hashpipe -I $i -o BINDHOST="enp64s0f0" -o BINDPORT=$(expr 60000 + $i) -o GPU=$(expr 0 + $i) -o OUTDIR="/buff0/gpu$i/" -m $((0x038 << $(expr $i \* 3))) -o WnFIL="weights_16x64x192.bin" -c $(expr 3 + $i \* 3) hrbf_net_thread -c $(expr 4 + $i \* 3) hrbf_gpu_thread -c $(expr 5 + $i \* 3) hrbf_output_thread &
    done
    ```
    In here, "hrbf_hashpipe" as plugin is launched by hashpipe software and created an instance of "-I 0". "-o BINDHOST="enp64s0f0" is used to bind the host with a given NIC name. "-o BINDPORT=$(expr 60000 + $i)" is used to bind the port number. "-o GPU=$(expr 0 + $i)" is used to bind the GPU core. "-o OUTDIR="/buff0/gpu$i/" is used to set the directory of the output files. "-m $((0x038 << $(expr $i \* 3)))" is used to set the mask of CPUs. "-o WnFIL='weights_16x64x192.bin'" is used to set the weights file name. "-c $(expr 3 +$i \* 3) hrbf_net_thread" is used to assign the CPU # for hrbf_net_thread, and so on.
* To check the run time status of this software, you can run following command:
    ```
    hashpipe_status_monitor.rb
    ```
* To start to receive the packets, you have to start a redis server and set the start_flag to 1:
    ```
    redis_cli -h host_ip
    set start_flag 1
    ```

### Settings
* Select to storage raw beamformed data
	set RECORD_BF_RAW to 1 in gpu_beamformer.h
* Select to process dual polarization signal
	set N_POLS to 2 in gpu_beamformer.h
* Select to transpose data before execute cuBblasCgemmBatched() 
	set TRANS_INPUT to 1 in gpu_beamformer.h 
* Select to receive packets from generated fake data 
	set FAKED_INPUT to 1 in gpu_beamformer.h 
* Set to debug mode and print testing information 
	set TEST_MODE to 1 hrbf_databuf.h 
* Set number of frequency bins, time samples, beams, elements, accumulation times 
	set N_FBIN, N_TSAMP, N_BEAM, N_ELEM, N_TSAMP_ACCU in gpu_beamformer.h 
