/*
 * gpu_beamformer.h
 *
 *  Created on: Aug 1, 2021
 *      Author: peix
 */

#ifndef GPU_BEAMFORMER_H_
#define GPU_BEAMFORMER_H_
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <cuComplex.h>
//#include "hrbf_databuf.h"

#define GPU_TEST_PRINT	1 // 1,print time consume information
#define FAKED_INPUT 0 // whether the input from faked data
#define TRANS_INPUT 1 // whether transpose input data
#define RECORD_BF_RAW 0 // whether record raw beamformed data
// Number of frequency bins, integer multiples of BLOCK_COLS = 8
#define N_FBIN		64
// Number of time samples, integer multiples of TILE_DIM = 32
#define N_TSAMP		(1024*64) //(40960*4) 1024 x 1MHz chunks,8us/frame, 128*1024=1s
// Number of formed beams, integer multiples of BLOCK_ROWS = 4
#define N_BEAM		1
// Number of elements, integer multiples of BLOCK_ROWS = 4
#define N_ELEM		8
// Number of time samples for each short time integration
#define N_TSAMP_ACCU		(1024*64) //(320*4)

#define N_POLS	1 // Number of polarizations
#define N_BEAM_STOKES (N_BEAM/N_POLS) // Number of beams after Stokes calculation
#define N_STOKES	1 // Number of Stokes items
#define N_ACCU	   (N_TSAMP/N_TSAMP_ACCU) // Number of short time integrations
#define N_INPUTS     (N_ELEM*N_FBIN*N_TSAMP) // Number of complex samples to process
#define N_WEIGHTS  (N_ELEM*N_FBIN*N_BEAM) // Number of complex beamformer weights
#define N_OUTPUTS_BF  (N_BEAM*N_TSAMP*N_FBIN) // Number of complex samples in beamformed output structure
#define N_OUTPUTS  (N_BEAM_STOKES*N_ACCU*N_FBIN*N_STOKES) // Number of samples in accumulator output structure

#define N_ELEM_PER_PACK		8			//Number of elements per packet
#define N_FBIN_PER_PACK		64			//Number of frequency channels per packet
#define N_TSAMP_PER_PACK	8			//Number of time samples per packet
#define N_PACK_PER_TSAMP	(N_TSAMP/N_TSAMP_PER_PACK)
#define N_PACK_PER_ELEM		(N_ELEM/N_ELEM_PER_PACK)

#ifdef __cplusplus
extern "C" {
#endif
/*void bf_get_offsets(float * offsets);
void bf_get_cal_filename(char * cal_filename);
void bf_get_algorithm(char * algorithm);
void bf_get_weight_filename(char * weight_filename);
long long unsigned int bf_get_xid();*/
void loadWeights(char * filename);
void initBeamformer(int cuda_core);
void bfCleanup();
//void runBeamformer(signed char * data_in, float * data_out);
void runBeamformer();
//void runBeamformer(signed char * data_in);
//extern signed char *h_data_r;
extern float * h_accu_stokes;
extern signed char *d_idata_r;
extern float *d_accu_stokes;
#ifdef __cplusplus
}
#endif

#endif /* GPU_BEAMFORMER_H_ */
