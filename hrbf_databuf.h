#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"
#include "gpu_beamformer.h"

#define CACHE_ALIGNMENT         64
#define N_INPUT_BLOCKS          3 
#define N_OUTPUT_BLOCKS         3

#define TEST_MODE	0 // 1,print test information
#define N_BYTES_HEADER		40			//Number of Bytes of header FID(4bit) | BID(6bit) | mcnt(54bit)
//#define N_BYTES_HEADER		8			//Number of Bytes of header FID(4bit) | BID(6bit) | mcnt(54bit)
#define DATA_SIZE_PER_PKT	(2*N_ELEM_PER_PACK*N_FBIN_PER_PACK*N_TSAMP_PER_PACK) //Packet size without Header,8192
#define PKTSIZE				(DATA_SIZE_PER_PKT+N_BYTES_HEADER)	//Total Packet size,8200
#define N_PKTS_PER_BUF		(N_INPUTS*2/DATA_SIZE_PER_PKT)	// Number of packets per data buffer,12*256=3072
#define BUFF_SIZE		(unsigned long)(N_INPUTS*2) 		//Buffer size with polarizations,4096x192x16x2
#define N_BYTES_PER_FILE	(N_OUTPUTS*4096*256*sizeof(float)) // Number of bytes per file to save accumulated Stokes(I,Q,U,V) parameter
#define N_FENGINE			(N_ELEM/N_ELEM_PER_PACK) //Number of F-engines,12
#define N_PACKETS_PER_SPEC	N_FENGINE			//Number of packets per spectrum,12
// provide packet offset in a buffer
#define hrbf_input_databuf_idx(mc,fid) (((mc*N_FENGINE+fid)*DATA_SIZE_PER_PKT)*sizeof(char))
/*#define N_BEAM			19
#define N_POST_VACC		1			//number of post vaccumulation, how many spectrums added together
#define N_POST_CHANS_COMB	1			//Number of post channels combining,how many channels added together, this number must be set to 2^n
#define N_CHAN_PER_PACK		2048			//Number of channels per packet
#define N_PACKETS_PER_SPEC	2			//Number of packets per spectrum
#define N_BYTES_DATA_POINT	1			//Number of bytes per datapoint
#define N_POLS_PKT		2			//Number of polarizations per packet
#define N_BYTES_HEADER		8			//Number of Bytes of header
#define N_SPEC_BUFF             (2*512*N_POST_VACC)		//Number of spectrums per buffer
#define N_BITS_DATA_POINT       (N_BYTES_DATA_POINT*8) 	//Number of bits per datapoint in packet
#define N_CHANS_SPEC		(N_CHAN_PER_PACK * N_PACKETS_PER_SPEC) 					//Channels in spectrum for 1 pole.
#define N_POST_CHANS_SPEC	(N_CHANS_SPEC/N_POST_CHANS_COMB) 					//number of channels after post channel merge, for filterbank file
#define DATA_SIZE_PACK		(unsigned long)(N_CHAN_PER_PACK * N_POLS_PKT *  N_BYTES_DATA_POINT) 	//Packet size without Header 
#define PKTSIZE			(DATA_SIZE_PACK + N_BYTES_HEADER)					//Total Packet size 
#define N_BYTES_PER_SPEC	(DATA_SIZE_PACK*N_PACKETS_PER_SPEC)					//Spectrum size with polarations
#define BUFF_SIZE		(unsigned long)(N_SPEC_BUFF*N_BYTES_PER_SPEC) 				//Buffer size with polarations
#define N_CHANS_BUFF		(N_SPEC_BUFF*N_CHANS_SPEC)     						//Channels in one buffer without polarations
//#define N_SPEC_PER_FILE		1199616/4 			// Number of spectrums per file \
				int{time(s)/T_samp(s)/N_SPEC_BUFF}*N_SPEC_BUFF  e.g. 20s data: int(20/0.001/128)*128
//#define N_BYTES_PER_FILE	(N_SPEC_PER_FILE * N_BYTES_PER_SPEC / N_POLS_PKT) 			// we can save (I,Q,U,V) polaration into disk. 
*/
/**************************** parameters for filterbank header ***********************/
//#define SAMP_TIME		64e-6			// sec, when acc_len=8
/*#define CLOCK			1000			// MHz
#define START_FREQ		1000			// MHz
#define FFT_CHANS		4096			// MHz, number of FFT channels in ROACH2
#define FREQ_RES		(CLOCK/2.0/FFT_CHANS)	// MHz
#define F_OFF   		(-1*FREQ_RES*N_POST_CHANS_COMB) // MHz
#define F_CH1   		(START_FREQ+CLOCK/2.0-FREQ_RES/2.0)	// 
#define ACC_LEN			32			// accumulation length defined in ROACH2
#define SAMP_TIME		(FFT_CHANS*2.0*ACC_LEN/CLOCK*1.0e-6)			// sec, when acc_len=32

#define FIL_LEN			60			// sec
#define N_BYTES_PER_FILE	(FIL_LEN/SAMP_TIME/N_POST_VACC*N_CHANS_SPEC/N_POST_CHANS_COMB) 			// we can save (I,Q,U,V) polaration into disk. 
*/


// Used to pad after hashpipe_databuf_t to maintain cache alignment
typedef uint8_t hashpipe_databuf_cache_alignment[
  CACHE_ALIGNMENT - (sizeof(hashpipe_databuf_t)%CACHE_ALIGNMENT)
];

/* INPUT BUFFER STRUCTURES*/
typedef struct hrbf_input_block_header {
   uint64_t	netmcnt;        // Counter for ring buffer
   		
} hrbf_input_block_header_t;

typedef uint8_t hrbf_input_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(hrbf_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct hrbf_input_block {

   hrbf_input_block_header_t header;
   hrbf_input_header_cache_alignment padding; // Maintain cache alignment
   signed char  data[BUFF_SIZE]; //Input buffer for all channels

} hrbf_input_block_t;

typedef struct hrbf_input_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   hrbf_input_block_t block[N_INPUT_BLOCKS];
} hrbf_input_databuf_t;


/*
  * OUTPUT BUFFER STRUCTURES
  */
typedef struct hrbf_output_block_header {

} hrbf_output_block_header_t;

typedef uint8_t hrbf_output_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(hrbf_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct hrbf_output_block {

   hrbf_output_block_header_t header;
   hrbf_output_header_cache_alignment padding; // Maintain cache alignment
   float  data[N_OUTPUTS]; //Output buffer for all channels

} hrbf_output_block_t;

typedef struct hrbf_output_databuf {
   hashpipe_databuf_t header;
   hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   hrbf_output_block_t block[N_OUTPUT_BLOCKS];
} hrbf_output_databuf_t;

/*
 * INPUT BUFFER FUNCTIONS
 */
hashpipe_databuf_t *hrbf_input_databuf_create(int instance_id, int databuf_id);

static inline hrbf_input_databuf_t *hrbf_input_databuf_attach(int instance_id, int databuf_id)
{
    return (hrbf_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hrbf_input_databuf_detach(hrbf_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void hrbf_input_databuf_clear(hrbf_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int hrbf_input_databuf_block_status(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_total_status(hrbf_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int hrbf_input_databuf_wait_free(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_busywait_free(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_wait_filled(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_busywait_filled(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_set_free(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_input_databuf_set_filled(hrbf_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *hrbf_output_databuf_create(int instance_id, int databuf_id);

static inline void hrbf_output_databuf_clear(hrbf_output_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline hrbf_output_databuf_t *hrbf_output_databuf_attach(int instance_id, int databuf_id)
{
    return (hrbf_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int hrbf_output_databuf_detach(hrbf_output_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int hrbf_output_databuf_block_status(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_output_databuf_total_status(hrbf_output_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int hrbf_output_databuf_wait_free(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_output_databuf_busywait_free(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}
static inline int hrbf_output_databuf_wait_filled(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_output_databuf_busywait_filled(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_output_databuf_set_free(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int hrbf_output_databuf_set_filled(hrbf_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}


