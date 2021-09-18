# hrbf_Pipeline_Hashpipe
### Introduction
    
* This  code is used to receive the packets from high speed ethernet, calculate the Stokes parameters and save the data to disc in filterbank format.The data will be stored in a temporary place which mounted on CPU RAM. A FRB real time search software known as Heimdall will look for the new *.fil file in a given directory, once Heimdall found interested signals the Filterbank files will be moved to disk, otherwise remove these raw data.<br>
* Three threads have developed to perform packet receiving, Stokes calculation and Filterbank data formatting, which are:
    * hrbf_net_thread; (packet receving)
    * hrbf_gpu_thread; (Stokes calculation,"gpu" is just a tradition name, no GPU used in here.)
    * hrbf_output_thread; (Filterbank data saving)

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
    Inside this shell script, only one command line will be executed see follows:
    ```
    hashpipe -p hrbf_hashpipe -I 0 -o BINDHOST="10.10.12.2" -c 1 hrbf_net_thread -c 2 hrbf_gpu_thread -c 3 hrbf_output_thread
    ```
    In here, "hrbf_hashpipe" as plugin was launched by hashpipe software and created an instance of "-I 0". "-o BINDHOST="10.10.12.2" is used to bind the host with a given IP address. "-c 1 hrbf_net_thread" is used to assign the CPU 1 for hrbf_net_thread, and so on.
* To check the run time status of this software, you can run following command:
    ```
    hashpipe_status_monitor.rb
    ```
### Settings
* Collapse data in time
	set N_POST_VACC in hrbf_databuf.h to perform post vaccumulation, add given number of spectrums together
* Collapse data in spectrum
	set N_POST_CHANS_COMB in hrbf_databuf.h to combine given number of channels together, this value must be set to 2^n
* Roaches concurrent data collection mechanism 
	use Redis database to set a start flag, check the value of flag to start collecting data. 
