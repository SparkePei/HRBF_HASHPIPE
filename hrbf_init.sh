#!/bin/bash
#######topology##############
#Ethernet interfaces to CPUs:
#enp64s0f0: 0-15,32-47
#enp64s0f1: 0-15,32-47
#enp96s0f0: 0-15,32-47
#enp96s0f1: 0-15,32-47
#
#GPUs to CPUs:
#gpu0000:1d:00.0: 0-15,32-47
#gpu0000:1e:00.0: 0-15,32-47
#gpu0000:3f:00.0: 0-15,32-47
#gpu0000:41:00.0: 0-15,32-47

# Remove old semaphore
echo removing old semaphore, if any
#rm /dev/shm/sem.hrbf_gpu_sem_device_*
sudo rm /dev/shm/sem.home_peix_hashpipe_status*
for i in 0 1 2 3 
	do
	echo hashpipe -p ./hrbf_hashpipe -I $i -o BINDHOST="enp64s0f0" -o BINDPORT=$(expr 60000 + $i) -o GPU=$(expr 0 + $i) -o OUTDIR="/buff0/gpu$i/" -m $((0x038 << $(expr $i \* 3))) -o WnFIL="weights_16x64x192.bin" -c $(expr 3 + $i \* 3) hrbf_net_thread -c $(expr 4 + $i \* 3) hrbf_gpu_thread -c $(expr 5 + $i \* 3) hrbf_output_thread \&
	hashpipe -p ./hrbf_hashpipe -I $i -o BINDHOST="enp64s0f0" -o BINDPORT=$(expr 60000 + $i) -o GPU=$(expr 0 + $i) -o OUTDIR="/buff0/gpu$i/" -m $((0x038 << $(expr $i \* 3))) -o WnFIL="weights_16x64x192.bin" -c $(expr 3 + $i \* 3) hrbf_net_thread -c $(expr 4 + $i \* 3) hrbf_gpu_thread -c $(expr 5 + $i \* 3) hrbf_output_thread &
done
