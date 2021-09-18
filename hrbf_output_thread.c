/*
 * hrbf_output_thread.c
 * 
 */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include "hashpipe.h"
#include "hrbf_databuf.h"
//#include "filterbank.h"
#include <sys/time.h>
#include "gpu_beamformer.h"

extern bool start_file_record;
extern double net_MJD;
//extern char dir_out[128];

static void *run(hashpipe_thread_args_t * args)
{
	printf("\nOutput file size is set to %f Mbytes.\n ",(float)N_BYTES_PER_FILE/1024/1024);

	//printf("\n%d Channels per Buff.\n ",N_CHANS_BUFF/N_POST_VACC/N_POST_CHANS_COMB);
	// Local aliases to shorten access to args fields
	// Our input buffer happens to be a hrbf_ouput_databuf
	hrbf_output_databuf_t *db = (hrbf_output_databuf_t *)args->ibuf;
	hashpipe_status_t st = args->st;
	const char * status_key = args->thread_desc->skey;
	int rv;
	int N_files=0;
	int block_idx = 0;
	uint64_t n_bytes_saved = 0;
	uint64_t n_bytes_per_file = N_BYTES_PER_FILE;
	int fil_full = 0;
	char dir_out[128];
	FILE * f_output;
	char fn_output[256];
        hashpipe_status_lock_safe(&st);
        hgets(st.buf, "OUTDIR", 80, dir_out);
        hputs(st.buf, "OUTDIR",dir_out);
        hashpipe_status_unlock_safe(&st);

	sleep(1);
	/* Main loop */
	while (run_threads()) {
		hashpipe_status_lock_safe(&st);
		hputi4(st.buf, "OUTBLKIN", block_idx);
		hputi8(st.buf, "DATSAVMB",(n_bytes_saved/1024/1024));
		hputi4(st.buf, "NFILESAV",N_files);
		hputs(st.buf, status_key, "waiting");
		hashpipe_status_unlock_safe(&st);

		// Wait for data to storage
		while ((rv=hrbf_output_databuf_wait_filled(db, block_idx))
		!= HASHPIPE_OK) {
		if (rv==HASHPIPE_TIMEOUT) {
			hashpipe_status_lock_safe(&st);
			hputs(st.buf, status_key, "blocked");
			hputi4(st.buf, "OUTBLKIN", block_idx);
			hashpipe_status_unlock_safe(&st);
			continue;
			} else {
				hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
				pthread_exit(NULL);
				break;
			}
		}
		
		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "processing");
		hputi4(st.buf, "OUTBLKIN", block_idx);
		hashpipe_status_unlock_safe(&st);
		if (fil_full ==0 && start_file_record ==1 ){
			struct tm  *now;
			time_t rawtime;
			//extern
			//char dir_out[128] = "/buff0/gpu0/";
			printf("Output file saved directory is: %s\n",dir_out);
			//char P[4] = {'I','Q','U','V'};
			printf("\n\nOpen new output file...\n\n");

			//char File_dir[] = "/mnt/fast_frb_data/B";
			char t_stamp[50];
	        time(&rawtime);
			now = localtime(&rawtime);
		    strftime(t_stamp,sizeof(t_stamp), "_%Y-%m-%d_%H-%M-%S",now);
		    printf("Time stamp is: %s\n",t_stamp);
	        sprintf(fn_output,"%s%s%d%s%d%s%d%s%s" ,dir_out,"output_",N_FBIN,"x",N_BEAM_STOKES,"x",N_ACCU,t_stamp,".bin");///home/peix/workspace/paf_sim/output_**.bin
	        printf("Start write data to %s\n",fn_output);
	        f_output = fopen(fn_output, "wb");

			//WriteHeader(f_fil_P1,net_MJD);
			//printf("write header done!\n");
			N_files += 1;

		}

		// write data to file
		fwrite(db->block[block_idx].data, sizeof(float), N_OUTPUTS, f_output);
		n_bytes_saved += N_OUTPUTS*sizeof(float);
	
		if (TEST_MODE){

			printf("**Save Information**\n");
			//printf("beam_ID:%d \n",beam_ID);
			printf("Buffsize: %lu",BUFF_SIZE);
			printf("File full:%d\n",fil_full);
			printf("Data save:%f MHz\n",(float)n_bytes_saved/1024/1024);
			printf("Total file size:%f MHz\n",(float)n_bytes_per_file/1024/1024);
			printf("Devide:%lu\n\n",n_bytes_saved % n_bytes_per_file);
		}
		// if saved data is reach the setting file size
		if (n_bytes_saved >= n_bytes_per_file){

			fil_full = 0;
			n_bytes_saved = 0;
			fclose(f_output);
		}
		else{
			fil_full = 1;
		}

		hrbf_output_databuf_set_free(db,block_idx);
		block_idx = (block_idx + 1) % db->header.n_block;

		//Will exit if thread has been cancelled
		pthread_testcancel();
	}
	return THREAD_OK;
}

static hashpipe_thread_desc_t hrbf_output_thread = {
	name: "hrbf_output_thread",
	skey: "OUTSTAT",
	init: NULL, 
	run:  run,
	ibuf_desc: {hrbf_output_databuf_create},
	obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor()
{
	register_hashpipe_thread(&hrbf_output_thread);
}

