/*hrbf_gpu_thread.c
 *
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include "hashpipe.h"
#include "hrbf_databuf.h"
#include "math.h"
#include "gpu_beamformer.h"

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

static const char * status_key;
//extern bool data_type;

typedef struct {
    bool    initialized;
    int     out_block_idx;
    int 	in_block_idx;
} calc_block_info_t;

static inline void initialize_block_info(calc_block_info_t * binfo)
{
    // If this block_info structure has already been initialized
    if(binfo->initialized) {
        return;
    }

    binfo->in_block_idx     = 0;
    binfo->out_block_idx    = 0;
    binfo->initialized	    = 1;
}
static calc_block_info_t binfo;

static void *run(hashpipe_thread_args_t * args)
{
    // Local aliases to shorten access to args fields
    hrbf_input_databuf_t *db_in = (hrbf_input_databuf_t *)args->ibuf;
    hrbf_output_databuf_t *db_out = (hrbf_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    status_key = args->thread_desc->skey;
    //polar_data_t data;
    int rv;
    int cuda_core;
    //char *weight_file;
    char weight_file[256];
    //char *bf_in;
    //float *bf_out;
    clock_t begin,end;
    double time_spent;
    /*if(!binfo.initialized) {
        initialize_block_info(&binfo);
    }*/
    initialize_block_info(&binfo);
	hashpipe_status_lock_safe(&st);
    hgeti4(st.buf, "GPU", &cuda_core); // must give from hashpipe run command
    hputi4(st.buf, "GPU", cuda_core);
    hgets(st.buf, "WnFIL", 80, weight_file);
    hputs(st.buf, "WnFIL",weight_file);
    hashpipe_status_unlock_safe(&st);

    // pin the databufs from cuda's point of view
    //cudaHostRegister((void *) db_in, sizeof(hrbf_input_databuf_t), cudaHostRegisterMapped); //zero copy,map allocation into device memory
    //cudaHostRegister((void *) db_in, sizeof(hrbf_input_databuf_t), cudaHostRegisterPortable); //pinned memory accessible by all cuda context
    cudaHostRegister((void *) db_in, sizeof(hrbf_input_databuf_t), cudaHostRegisterDefault); //page-locked allocation flag
    //cudaHostRegister((void *) db_out, sizeof(hrbf_output_databuf_t), cudaHostRegisterMapped);
    //cudaHostAlloc(&h_data_r,2*N_INPUTS*sizeof(signed char),cudaHostAllocMapped);
	// initialize beamformer, allocate memory
	begin = clock();
	initBeamformer(cuda_core);
	//bf_out = (float *)malloc(N_OUTPUTS*sizeof(float));
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("N_BEAM: %d N_ELEM: %d N_TSAMP: %d N_FBIN: %d N_TSAMP_ACCU: %d\n",N_BEAM,N_ELEM,N_TSAMP,N_FBIN,N_TSAMP_ACCU);
		printf("Initialize beamformer elapsed: %3.3f ms\n",time_spent);
	}
	// read weights from file
	begin = clock();
	loadWeights(weight_file);
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Load weights from %s elapsed: %3.3f ms\n",weight_file,time_spent);
	}

    while (run_threads()) {
    	hashpipe_status_lock_safe(&st);
        hputi4(st.buf, "GPU-IN", binfo.in_block_idx);
        hputs(st.buf, status_key, "waiting");
        hputi4(st.buf, "GPU-OUT", binfo.out_block_idx);
        hashpipe_status_unlock_safe(&st);


        // Wait for new input block to be filled
        while ((rv=hrbf_input_databuf_wait_filled(db_in, binfo.in_block_idx)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "blocked");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
                pthread_exit(NULL);
                break;
            }
        }
        // Note processing status
        hashpipe_status_lock_safe(&st);
        hputs(st.buf, status_key, "processing");
        hashpipe_status_unlock_safe(&st);

        // Copy input data buffer to pointer
        // memcpy(&db_out->block[binfo.out_block_idx].data,bf_out,N_OUTPUTS*sizeof(float));
        // run beamformer
        /*for (int i=0;i<128;i++){
        	for (int j=0;j<4;j++){
        		for (int k=0;k<4;k++){
        			printf("%d ",db_in->block[binfo.out_block_idx].data[i*N_ELEM*N_FBIN*2+j*2*N_FBIN+k]);
        		}
        	}

        }*/
        /*signed char data;
		printf("GPU thread check the first 128 data point in data buffer:\n");
		for (int i=0;i<256;i++){
			for (int j=0;j<16;j++){
				data = *(signed char *)(db_in->block[binfo.in_block_idx].data+8192*12*i+8192/16*j+1);
				printf("%d ",data);
			}
		}*/

		/*for (int i=0;i<4096;i++){
			data = *(signed char *)(db_in->block[binfo.in_block_idx].data+16*2*192*i+1);
			printf("%d ",data);
		}
		printf("\n");*/
        //h_data_r = db_in->block[binfo.out_block_idx].data;
        begin = clock();
        cudaDeviceSynchronize();
        cudaMemcpy(d_idata_r,db_in->block[binfo.in_block_idx].data,N_INPUTS*2*sizeof(signed char),cudaMemcpyHostToDevice);
        //cudaHostGetDevicePointer((void **)&d_idata_r, (void *)db_in->block[binfo.out_block_idx].data, 0);
        cudaDeviceSynchronize();
    	if (GPU_TEST_PRINT) {
    		end = clock();
    		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
    		printf("GPU thread copy data from input data buffer elapsed: %3.3f ms\n",time_spent);
    	}
    	begin = clock();

        runBeamformer();
        //runBeamformer(db_in->block[binfo.out_block_idx].data);
        end = clock();
        time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
        printf("Run beamformer totally elapsed: %3.3f ms\n",time_spent);

        /*for (int i=0;i<N_FBIN;i++){
                	for (int j=0;j<N_BEAM_STOKES*N_STOKES;j++){
                		if(j<4){
                			printf("%3.2f ",h_accu_stokes[i*N_BEAM_STOKES*N_STOKES*N_ACCU+j*N_ACCU]);
                		}
                	}
                }*/
        /*runBeamformer(db_in->block[binfo.out_block_idx].data, bf_out);
        for (int i=0;i<N_FBIN;i++){
        	for (int j=0;j<N_BEAM_STOKES*N_STOKES;j++){
        		if(j<4){
        			printf("%3.2f ",bf_out[i*N_BEAM_STOKES*N_STOKES*N_ACCU+j*N_ACCU]);
        		}
        	}
        }*/
        // Mark input block as free and advance
        hrbf_input_databuf_set_free(db_in, binfo.in_block_idx);
        binfo.in_block_idx = (binfo.in_block_idx + 1) % db_in->header.n_block;

        // Wait for new output block to be free
        while ((rv=hrbf_output_databuf_wait_free(db_out, binfo.out_block_idx)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "block_out");
                hashpipe_status_unlock_safe(&st);
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                pthread_exit(NULL);
                break;
            }
        }
        /*Copy beamformed result to output data buffer*/
        begin = clock();
        //cudaHostGetDevicePointer((void **)&d_accu_stokes, (void *)db_out->block[binfo.out_block_idx].data, 0);
        memcpy(&db_out->block[binfo.out_block_idx].data,h_accu_stokes,N_OUTPUTS*sizeof(float));
        /*for (int i=0;i<N_FBIN;i++){
        	for (int j=0;j<N_BEAM_STOKES*N_STOKES;j++){
        		if(j<4){
        			printf("%3.2f ",db_out->block[binfo.out_block_idx].data[i*N_BEAM_STOKES*N_STOKES*N_ACCU+j*N_ACCU]);
        		}
        	}
        }
        printf("\n");*/
        if (GPU_TEST_PRINT) {
			end = clock();
			time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
			printf("GPU thread copy data to output data buffer elapsed: %3.3f ms\n",time_spent);
        }
        if (TEST_MODE){
			fprintf(stderr,"**Net tread**\n");
			fprintf(stderr,"wait for output writting..\n");
			//fprintf(stderr,"Data size: %lu \n\n",sizeof(bf_out));
		}
        // Mark output block as full and advance
        hrbf_output_databuf_set_filled(db_out,binfo.out_block_idx);
        binfo.out_block_idx = (binfo.out_block_idx + 1) % db_out->header.n_block;		
        /* Check for cancel */
        pthread_testcancel();
    }
	// unpin the databufs from cuda's point of view
    cudaHostUnregister((void *) db_in);
    //cudaHostUnregister((void *) db_out);

	// Free resources
	bfCleanup();
    return THREAD_OK;
}

static hashpipe_thread_desc_t hrbf_gpu_thread = {
    name: "hrbf_gpu_thread",
    skey: "BFSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {hrbf_input_databuf_create},
    obuf_desc: {hrbf_output_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hrbf_gpu_thread);
}

