/*
 * hrbf_net_thread.c
 *
 * This allows you to receive pakets from local ethernet, unpack the packets, and then write data into a shared memory buffer. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <smmintrin.h>
#include <immintrin.h>
#include "hashpipe.h"
#include "hrbf_databuf.h"
#include <hiredis/hiredis.h>
#include "gpu_beamformer.h"

//using namespace std;

#define PKTSOCK_BYTES_PER_FRAME (16384) //(16384)
#define PKTSOCK_FRAMES_PER_BLOCK (80)
#define PKTSOCK_NBLOCKS (200)
#define PKTSOCK_NFRAMES (PKTSOCK_FRAMES_PER_BLOCK * PKTSOCK_NBLOCKS)

//double MJD;
//int     beam_ID;
//bool start_file=0;
bool start_file_record=0;
double net_MJD;
int bindport = 60001;
char bindhost[80];
static int total_packets_counted = 0;
static hashpipe_status_t *st_p;
static  const char * status_key;

typedef struct {
    uint64_t	mcnt;            // counter for packet
    uint8_t		fid;    // F-engine ID: 0 - 15
    uint8_t		bid;      // B-engine ID:0 - 63
} packet_header_t;

typedef struct {

    uint64_t 	miss_pkt;
    long   	offset;
    int     initialized;
    int		block_idx;
    int 	self_bid; 		// bf id of this node
    bool	start_flag;
    bool	first_frame;
    uint64_t mcnt_start;    // The starting mcnt for this block
    uint64_t 	cur_mcnt;
    int      mc;            // Indices for packet payload destination
    int		cur_mcnt_fid;		// count the number of frames from f-engines
    int		fid;            // Indices for packet payload destination
    long	pkt_size_count; // Count received packet size in a block
} block_info_t;

static block_info_t binfo;

// This function must be called once and only once per block_info structure!
// Subsequent calls are no-ops.
static inline void initialize_block_info(block_info_t * binfo)
{
    // If this block_info structure has already been initialized
    if(binfo->initialized) {
        return;
    }
    // Initialize our BID
    binfo->self_bid = 0;
    hashpipe_status_lock_safe(st_p);
    hgeti4(st_p->buf, "BID", &binfo->self_bid);
    hashpipe_status_unlock_safe(st_p);

    binfo->cur_mcnt		= 0;
    binfo->mcnt_start  	= 0;
    binfo->block_idx	= 0;
    binfo->start_flag	= 0;
    binfo->offset		= 0;
    binfo->miss_pkt		= 0;
    binfo->initialized	= 1;
    binfo->first_frame	= 1;
    binfo->pkt_size_count=0;
}

static int init(hashpipe_thread_args_t * args)
{
	// define network params
	//char bindhost[80];
	//int bindport = 60001;
	hashpipe_status_t st = args->st;
	strcpy(bindhost, "0.0.0.0");

	hashpipe_status_lock_safe(&st);
	// Get info from status buffer if present (no change if not present)
	hgets(st.buf, "BINDHOST", 80, bindhost);
	hgeti4(st.buf, "BINDPORT", &bindport);
	// Store bind host/port info etc in status buffer
	hputs(st.buf, "BINDHOST", bindhost);
	hputi4(st.buf, "BINDPORT", bindport);

	hputi8(st.buf, "NPACKETS", 0);
	hputi8(st.buf,"PKTLOSS",0);
	hputr8(st.buf,"LOSSRATE",0.0);
	hputu4(st.buf, "NETRECV",  0);
	hputu4(st.buf, "NETDROPS", 0);

	//hputi8(st.buf, "PKTSIZE", SIZE_OF_PKT);
	hashpipe_status_unlock_safe(&st);
	//initialize_block_info(&binfo);
	// Set up pktsock
	struct hashpipe_pktsock *p_ps = (struct hashpipe_pktsock *)
	malloc(sizeof(struct hashpipe_pktsock));
	if(!p_ps) {
		perror(__FUNCTION__);
		return -1;
	}
	/* Make frame_size be a divisor of block size so that frames will be
	contiguous in mapped memory.  block_size must also be a multiple of
	page_size.  Easiest way is to oversize the frames to be 16384 bytes, which
	is bigger than we need, but keeps things easy. */
	p_ps->frame_size = PKTSOCK_BYTES_PER_FRAME;
	// total number of frames
	p_ps->nframes = PKTSOCK_NFRAMES;
	// number of blocks
	p_ps->nblocks = PKTSOCK_NBLOCKS;
	int rv = hashpipe_pktsock_open(p_ps, bindhost, PACKET_RX_RING);
	if (rv!=HASHPIPE_OK) {
	hashpipe_error("hrbf_net_thread", "Error opening pktsock.");
	pthread_exit(NULL);
	}
	// Store packet socket pointer in args
	args->user_data = p_ps;
	// Success!
	printf("Net thread initialize success!");
	return 0;
}



static inline void get_header(unsigned char *p_frame, packet_header_t * pkt_header)
{
    uint32_t raw_header_mcnt;
    unsigned char beam_id;
//    raw_header = le64toh(*(unsigned long long *)p->data);
    //memcpy(&raw_header_mcnt,packet+12,4*sizeof(char));
    memcpy(&raw_header_mcnt,PKT_UDP_DATA(p_frame)+32,4*sizeof(char));
    memcpy(&beam_id,PKT_UDP_DATA(p_frame)+22,1*sizeof(char));
    pkt_header->mcnt        = raw_header_mcnt;
    //pkt_header->seq        = raw_header_mcnt;
    pkt_header->bid     = (int)beam_id;
    //beam_ID = (int)beam_id;
    if (TEST_MODE){
            fprintf(stderr,"**Header**\n");
            fprintf(stderr,"Mcnt of Header is :%ld \n ",pkt_header->mcnt);
            fprintf(stderr,"Beam ID is:%d\n\n",pkt_header->bid);
        }
}

/*    uint64_t raw_header;
    signed char data;
    //raw_header = le64toh(*(unsigned long long *)p->data);
    memcpy(&raw_header,PKT_UDP_DATA(p_frame),N_BYTES_HEADER*sizeof(char));
    //memcpy(&raw_header,p_frame,N_BYTES_HEADER*sizeof(char));
    //raw_header = *(unsigned long long *)PKT_UDP_DATA(p_frame);
    pkt_header->mcnt = raw_header  & 0x003fffffffffffff;
    pkt_header->fid = (raw_header >> 60)  & 0x000000000000000f; //0 - 15
    pkt_header->bid = (raw_header >> 54)  & 0x000000000000003f; //0 - 63
    printf("\n");
    if (TEST_MODE){
	    fprintf(stderr,"**Header**\n");
	    fprintf(stderr,"Mcnt of Header is :%lu \n",pkt_header->mcnt);
	    fprintf(stderr,"Raw Header: %lu \n",raw_header);
	    fprintf(stderr,"F-engine ID:%d\n",pkt_header->fid);
	    fprintf(stderr,"B-engine ID:%d\n",pkt_header->bid);
	    printf("p_frame size is: %d\n",PKT_UDP_SIZE(p_frame));
	}
}*/

// Calculate the buffer address for packet payload, verifies FID and BID of packets
static inline int calc_block_indices(block_info_t * binfo, packet_header_t * pkt_header) {
    // Verify FID and BID
	//printf("Verify FID and BID: ");
    if (pkt_header->fid >= N_FENGINE) {
        hashpipe_error(__FUNCTION__, "packet FID %u out of range (0-%d)", pkt_header->fid, N_FENGINE-1);
        return -1;
    }
    else if (pkt_header->bid != binfo->self_bid) {
        hashpipe_error(__FUNCTION__, "unexpected packet BID %d (expected %d)", pkt_header->bid, binfo->self_bid);
        return -1;
    }
    binfo->mc = pkt_header->mcnt % binfo->mcnt_start;
    binfo->fid = pkt_header->fid;
    //printf("Done!\n");
    return 0;
}

double UTC2JD(double year, double month, double day){
	double jd;
	double a;
	a = floor((14-month)/12);
	year = year+4800-a;
	month = month+12*a-3;
	jd = day + floor((153*month+2)/5)+365*year+floor(year/4)-floor(year/100)+floor(year/400)-32045;
	return jd;
}

static void *run(hashpipe_thread_args_t * args){
	hrbf_input_databuf_t *db  = (hrbf_input_databuf_t *)args->obuf;
	hashpipe_status_t st = args->st;
	status_key = args->thread_desc->skey;
	st_p = &st; // allow global (this source file) access to the status buffer

	int i, rv,input,n;
	uint64_t mcnt = 0;
	int block_idx = 0;
	uint64_t header; // 64 bit counter
	// unsigned char data_pkt[SIZE_OF_PKT]; // save received packet
	packet_header_t pkt_header;
	uint64_t SEQ=0;
	uint64_t LAST_SEQ=0;
	uint64_t pkt_loss=0; // number of packets has been lost
	//int first_pkt=1;
	double pkt_loss_rate; // packets lost rate
	unsigned int pktsock_pkts = 0;  // Stats counter from socket packet
	unsigned int pktsock_drops = 0; // Stats counter from socket packet

    uint64_t pkt_mcnt = 0;
    uint8_t	pkt_fid	= 0;

    if(!binfo.initialized) {
        initialize_block_info(&binfo);
        db->block[block_idx].header.netmcnt=0;
        printf("\nInitailized!\n");
    }

	double Year, Month, Day;
	double jd;
	time_t timep;
	struct tm *p;
	struct timeval currenttime;
	time(&timep);
	p=gmtime(&timep);
	Year=p->tm_year+1900;
	Month=p->tm_mon+1;
	Day=p->tm_mday;
	jd = UTC2JD(Year, Month, Day); 
	net_MJD=jd+(double)((p->tm_hour-12)/24.0)
                               +(double)(p->tm_min/1440.0)
                               +(double)(p->tm_sec/86400.0)
                               +(double)(currenttime.tv_usec/86400.0/1000000.0)
								-(double)2400000.5;
	printf("MJD time of packets is %lf\n",net_MJD);
    int pkt_size;
	uint64_t npackets = 0; //number of received packets

	redisContext *redis_c;
	redisReply *reply;
	const char *hostname = "xbd3";
	int redis_port = 6379;

	struct timeval timeout = { 1, 500000 }; // 1.5 seconds
	redis_c = redisConnectWithTimeout(hostname, redis_port, timeout);
	if (redis_c == NULL || redis_c->err) {
		if (redis_c) {
			printf("Connection error: %s\n", redis_c->errstr);
			redisFree(redis_c);
		} else {
			printf("Connection error: can't allocate redis context\n");
		}
		exit(1);
	}

	/* Give all the threads a chance to start before opening network socket */
	sleep(1);
	/* Get receiving flag from redis server */
	printf("waiting for set start_flag to 1 on server to start ...\n");
	do {
		reply = (redisReply *)redisCommand(redis_c,"GET start_flag");
		sleep(0.1);
	} while(strcmp(reply->str,"1")!=0); // if start_flag set to 1 then start data receiving.


	printf("GET value from %s and start_flag is: %s, start data receiving...\n", hostname, reply->str);
	freeReplyObject(reply);
	// wait until the integer time value changes.
	time_t time0=time(&timep);
	while(time(&timep)==time0);

	hashpipe_status_lock_safe(&st);
	// Get info from status buffer if present (no change if not present)
	hputs(st.buf, status_key, "running");
	hashpipe_status_unlock_safe(&st);

	// Get pktsock from args
	struct hashpipe_pktsock * p_ps = (struct hashpipe_pktsock*)args->user_data;
	pthread_cleanup_push(free, p_ps);
	pthread_cleanup_push((void (*)(void *))hashpipe_pktsock_close, p_ps);

	// Drop all packets to date
	unsigned char *p_frame;
	while(p_frame=hashpipe_pktsock_recv_frame_nonblock(p_ps)) {
		hashpipe_pktsock_release_frame(p_frame);
	}

	start_file_record  = 1;
	// Main loop
	while (run_threads()){		
		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "waiting");
		hputi4(st.buf, "NETBKOUT", block_idx);
		hputi8(st.buf,"NETMCNT",mcnt);
		hashpipe_status_unlock_safe(&st);

		// Wait for data
		/* Wait for new block to be free, then clear it
		 * if necessary and fill its header with new values.
		 */
		while ((rv=hrbf_input_databuf_wait_free(db, block_idx)) 
			    != HASHPIPE_OK) {
			if (rv==HASHPIPE_TIMEOUT) {
			    hashpipe_status_lock_safe(&st);
			    hputs(st.buf, status_key, "blocked");
			    hashpipe_status_unlock_safe(&st);
			    continue;
			} else {
			    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
			    pthread_exit(NULL);
			    break;
			}
		}

		hashpipe_status_lock_safe(&st);
		hputs(st.buf, status_key, "receiving");
		hashpipe_status_unlock_safe(&st);
		//
		// receiving packets
		for(int i=0;i<N_PKTS_PER_BUF;i++){
			do {
				p_frame = hashpipe_pktsock_recv_udp_frame_nonblock(p_ps, bindport);
			} 
			while (!p_frame && run_threads());
			//while (!pkt_size && run_threads());
			// check the first ten data point
			/*signed char data;
			for (int i=0;i<10;i++){
				data = *(signed char *)PKT_UDP_DATA(p_frame+8+i);
				printf("%d ",data);
			}*/

			if(!run_threads()) break;
			// Parse packet header
			get_header(p_frame,&pkt_header);
			SEQ = pkt_header.mcnt;
			//SEQ = pkt_header.mcnt*N_PACKETS_PER_SPEC+pkt_header.fid;
			//printf("SEQ is: %ld\n",SEQ);
			//pkt_fid	= pkt_header.fid;
 			if(npackets == 0){
				LAST_SEQ = (SEQ-1);
				//LAST_SEQ = pkt_header.mcnt*N_PACKETS_PER_SPEC-1;
 			}
 			if(binfo.first_frame){
 				binfo.mcnt_start = pkt_header.mcnt;
 				printf("mcnt_start is: %ld\n",binfo.mcnt_start);
 				binfo.first_frame = 0;
				binfo.pkt_size_count = 0;
		signed char data;
		/*printf("Net thread check the first 128 data point in data buffer:\n");
		for (int i=0;i<8192;i++){
			data = *(signed char *)PKT_UDP_DATA(p_frame+40+i);			
			printf("%d ",data);
			}*/
 			}
			npackets++;

		    binfo.mc = pkt_header.mcnt % binfo.mcnt_start;
		    //printf("mc is: %d\n",binfo.mc);
		if(binfo.mc >= N_PKTS_PER_BUF) {break;}
            /*if (binfo.cur_mcnt_fid == N_PACKETS_PER_SPEC){
            	binfo.cur_mcnt   += 1;
            	binfo.cur_mcnt_fid = 0;
            }*/
	        // Validate FID and XID
	        // Calculate "mc" and "fid" which index the buffer for writing packet payload
	        /*if (calc_block_indices(&binfo, &pkt_header) == -1) {
	            hashpipe_error(__FUNCTION__, "invalid FID and XID in header");
	            pthread_exit(NULL);
	        }*/
			hashpipe_status_lock_safe(&st);
			hputi8(st.buf, "NPACKETS", npackets);
			hashpipe_status_unlock_safe(&st);

	        //printf("calc block indices: done!\n");
	        // Copy data into buffer

            //binfo.offset = hrbf_input_databuf_idx(binfo.mc,pkt_fid);
            //cout << "offset is:" << binfo.offset << endl;
            //printf("offset is: %ld\n",binfo.offset);
memcpy(db->block[block_idx].data+binfo.pkt_size_count*DATA_SIZE_PER_PKT, PKT_UDP_DATA(p_frame)+40, DATA_SIZE_PER_PKT*sizeof(unsigned char));
//memcpy(db->block[block_idx].data+binfo.mc*DATA_SIZE_PER_PKT, PKT_UDP_DATA(p_frame)+40, DATA_SIZE_PER_PKT*sizeof(unsigned char));
binfo.pkt_size_count++;
            //memcpy((db->block[block_idx].data)+DATA_SIZE_PER_PKT, PKT_UDP_DATA(p_frame)+N_BYTES_HEADER, DATA_SIZE_PER_PKT*sizeof(signed char));
            //memcpy((db->block[block_idx].data)+binfo.offset, PKT_UDP_DATA(p_frame)+N_BYTES_HEADER, DATA_SIZE_PER_PKT*sizeof(signed char));
			hashpipe_pktsock_release_frame(p_frame);
			//if (binfo.offset >= (BUFF_SIZE-DATA_SIZE_PER_PKT)){break;}
			//memcpy(db->block[block_idx].data_in+i*DATA_SIZE_PER_PKT, PKT_UDP_DATA(p_frame)+8, DATA_SIZE_PER_PKT*sizeof(unsigned char));
			pthread_testcancel();
		}

		// check the first ten data point
		/*signed char data;
		printf("Net thread check the first 128 data point in data buffer:\n");
		for (int i=0;i<8192;i++){
			data = *(signed char *)(db->block[block_idx].data+8192*32*i);
			printf("%d ",data);
		}*/

		#ifdef TEST_MODE
			printf("number of lost packets is : %lu\n",pkt_loss);
		#endif
		// Handle variable packet size!
		int packet_size = PKT_UDP_SIZE(p_frame) - 8;
		#ifdef TEST_MODE
			printf("packet size is: %d\n",packet_size);
		#endif

		//SEQ = pkt_header.seq;
		//printf("SEQ is : %lu\n",SEQ);
		//pkt_loss += SEQ - (LAST_SEQ+N_PKTS_PER_BUF);
		if(SEQ > (LAST_SEQ+N_PKTS_PER_BUF)){
			binfo.miss_pkt += SEQ - (LAST_SEQ+N_PKTS_PER_BUF);}
		pkt_loss_rate = (double)binfo.miss_pkt/(double)npackets*100.0;
		LAST_SEQ = SEQ;
		binfo.first_frame = 1;
		binfo.pkt_size_count = 0;
		// Get stats from packet socket
		hashpipe_pktsock_stats(p_ps, &pktsock_pkts, &pktsock_drops);

		hashpipe_status_lock_safe(&st);
		hputi8(st.buf, "NPACKETS", npackets);
		hputi8(st.buf,"PKTLOSS",binfo.miss_pkt);
		hputr8(st.buf,"LOSSRATE",pkt_loss_rate);		
		hputu4(st.buf, "NETRECV",  pktsock_pkts);
		hputu4(st.buf, "NETDROPS", pktsock_drops);
		hashpipe_status_unlock_safe(&st);

		// Mark block as full
		if(hrbf_input_databuf_set_filled(db, block_idx) != HASHPIPE_OK) {
			hashpipe_error(__FUNCTION__, "error waiting for databuf filled call");
			pthread_exit(NULL);
		}

		db->block[block_idx].header.netmcnt = mcnt;
		block_idx = (block_idx + 1) % db->header.n_block;
		mcnt++;

		/* Will exit if thread has been cancelled */
		pthread_testcancel();
	}
	pthread_cleanup_pop(1); /* Closes push(hashpipe_pktsock_close) */
	pthread_cleanup_pop(1); /* Closes push(free) */
	// Thread success!
	return THREAD_OK;
}

static hashpipe_thread_desc_t hrbf_net_thread = {
	name: "hrbf_net_thread",
	skey: "NETSTAT",
	init: init,
	run:  run,
	ibuf_desc: {NULL},
	obuf_desc: {hrbf_input_databuf_create}
};

static __attribute__((constructor)) void ctor()
{
  register_hashpipe_thread(&hrbf_net_thread);
}
