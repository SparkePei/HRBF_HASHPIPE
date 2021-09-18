#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuComplex.h>
#include <time.h>
#include "gpu_beamformer.h"
//nvcc test_cublasCgemmBatched.cu -lcublas -o test_cublasCgemmBatched
//sudo nvidia-persistenced --user peix
using namespace std;

/*#define TRANS_INPUT 1 // whether transpose input data
#define RECORD_BF_RAW 0 // whether record raw beamformed data
// Number of frequency bins, integer multiples of BLOCK_COLS = 8
#define N_FBIN		16
// Number of time samples, integer multiples of TILE_DIM = 32
#define N_TSAMP		4096
// Number of formed beams, integer multiples of BLOCK_ROWS = 4
#define N_BEAM		64
// Number of elements, integer multiples of BLOCK_ROWS = 4
#define N_ELEM		192
// Number of time samples for each short time integration
#define N_TSAMP_ACCU		32
#define N_POLS	2 // Number of polarizations
#define N_BEAM_STOKES (N_BEAM/N_POLS) // Number of beams after Stokes calculation
#define N_STOKES	4 // Number of Stokes items
#define N_ACCU	   (N_TSAMP/N_TSAMP_ACCU) // Number of short time integrations
#define N_INPUTS     (N_ELEM*N_FBIN*N_TSAMP) // Number of complex samples to process
#define N_WEIGHTS  (N_ELEM*N_FBIN*N_BEAM) // Number of complex beamformer weights
#define N_OUTPUTS_BF  (N_BEAM*N_TSAMP*N_FBIN) // Number of complex samples in beamformed output structure
#define N_OUTPUTS  (N_BEAM_STOKES*N_ACCU*N_FBIN*N_STOKES) // Number of samples in accumulator output structure
*/
// For CUDA function related time consume test
cudaEvent_t     start, stop;
float   elapsedTime;
// For file I/O and CPU time consume test
clock_t begin,end,begin_main,end_main;
double time_spent;
// Three dimension of CUDA block, dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS)
//const int TILE_DIM = 16; //32
static int TILE_DIM = 16; //16
static int BLOCK_ROWS = 4; //4
static int BLOCK_COLS = 8; //8
dim3 dimGrid;
dim3 dimBlock(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
// Matrix dimension to be calculated by cuBLAS
int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C;
// handle to the cuBLAS library context
static cublasHandle_t handle;
cublasStatus_t stat;
cudaError_t cudaStat;
// Define variables on device
static cuComplex *d_weights;
static cuComplex *d_tdata;
static cuComplex *d_idata;
static cuComplex *d_net_data;
static cuComplex *d_beamformed;
static cuComplex **d_arr_A = NULL;
static cuComplex **d_arr_B = NULL;
cuComplex **d_arr_C = NULL;
float *d_weights_r;
signed char *d_idata_r;
float *d_stokes_out, *d_accu_stokes_in, *d_power_out;
float *d_accu_stokes;
// Define variables on host
static cuComplex **h_arr_A;
static cuComplex **h_arr_B;
static cuComplex **h_arr_C;
float *h_weights_r;
//signed char *h_data_r; // for file read
cuComplex *h_beamformed;
float *h_accu_stokes;
//CUDA_VISIBLE_DEVICES = 1;
// Define variables for directory and file name
//char dir[128] = "/home/peix/workspace/paf_sim/";
char dir[128] = "/buff0/";
char dir_output[128];
char fn_weight[256];
char fn_data[256];
char fn_output_bf[256];
char fn_output[256];
// CUDA device to be used
int cuda_core = 0;

void initBeamformer(int cuda_core){
	int rv = cudaSetDevice(cuda_core);
	printf("Set CUDA device to GPU#: %d\n",cuda_core);
	sprintf(dir_output,"%s%s%d%s" ,dir,"gpu",cuda_core,"/");
	// Creat cuda event and cublas handle
	cudaEventCreate( &start );
	cudaEventCreate( &stop ) ;
	cublasCreate(&handle);
	// Matrix dimension assignment
	nr_rows_A = N_BEAM;
	nr_cols_A = N_ELEM;
	nr_rows_B = N_ELEM;
	nr_cols_B = N_TSAMP;
	nr_rows_C = N_BEAM;
	// Allocate memory for weights
	cudaHostAlloc(&h_weights_r,2*N_WEIGHTS*sizeof(float),cudaHostAllocMapped);
	// Allocate memory for data
	//cudaHostAlloc(&h_data_r,2*N_INPUTS*sizeof(signed char),cudaHostAllocMapped);
	// Allocate memory for beamformed data
	cudaHostAlloc(&h_beamformed,N_OUTPUTS_BF*sizeof(cuComplex),cudaHostAllocMapped);

	// Allocate memory for accumulated data
	cudaMallocHost(&h_accu_stokes,N_OUTPUTS*sizeof(float));
	// Allocate memory to host arrays - This is all memory allocated to arrays that are used by gemmBatched. Allocate 3 arrays on CPU
	cudaHostAlloc((void **)&h_arr_A, nr_rows_A * nr_cols_A *N_FBIN*sizeof(cuComplex*),cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_arr_B, nr_rows_B * nr_cols_B *N_FBIN*sizeof(cuComplex*),cudaHostAllocMapped);
	cudaHostAlloc((void **)&h_arr_C, nr_rows_C * nr_cols_B *N_FBIN*sizeof(cuComplex*),cudaHostAllocMapped);
	// Allocate memory on GPU
	cudaMalloc(&d_weights, N_WEIGHTS*sizeof(cuComplex));
	cudaMalloc(&d_idata_r,2*nr_rows_B * nr_cols_B *N_FBIN*sizeof(signed char));
	cudaMalloc(&d_tdata, N_INPUTS*sizeof(cuComplex));
	cudaMalloc(&d_idata, N_INPUTS*sizeof(cuComplex));
	cudaMalloc(&d_net_data, N_INPUTS*sizeof(cuComplex));
	cudaMalloc(&d_beamformed,N_OUTPUTS_BF*sizeof(cuComplex));
	cudaMalloc(&d_stokes_out,N_OUTPUTS_BF*2*sizeof(float));
	cudaMalloc(&d_power_out,N_OUTPUTS_BF*sizeof(float));
	cudaMalloc(&d_accu_stokes_in,N_OUTPUTS_BF*2*sizeof(float));
	cudaMalloc(&d_accu_stokes,N_OUTPUTS*sizeof(float));
	// Allocate memory for each batch in an array.
	for(int i = 0; i < N_FBIN; i++){
		h_arr_A[i] = d_weights + i*nr_rows_A*nr_cols_A;
		h_arr_B[i] = d_tdata + i*nr_rows_B*nr_cols_B;
		h_arr_C[i] = d_beamformed + i*nr_rows_C*nr_cols_B;
	}
	// Get the device pointers for the pinned CPU memory mapped into the GPU memory, implement zero copy
	cudaStat = cudaHostGetDevicePointer((void **)&d_arr_A, (void *)h_arr_A, 0);
	assert(!cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **)&d_arr_B, (void *)h_arr_B, 0);
	assert(!cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **)&d_arr_C, (void *)h_arr_C, 0);
	assert(!cudaStat);
	//cudaMalloc(&d_weights_r,nr_rows_A * nr_cols_A *2*N_FBIN*sizeof(float));
	//cudaMemcpy(d_weights_r,h_weights_r,nr_rows_A * nr_cols_A *2*N_FBIN*sizeof(float),cudaMemcpyHostToDevice);
	cudaStat = cudaHostGetDevicePointer((void **)&d_weights_r, (void *)h_weights_r, 0);
	assert(!cudaStat);
	//cudaMalloc(&d_idata_r,2*nr_rows_B * nr_cols_B *N_FBIN*sizeof(signed char));
	//cudaMemcpy(d_idata_r,h_data_r,2*nr_rows_B * nr_cols_B *N_FBIN*sizeof(float),cudaMemcpyHostToDevice);
	//cudaStat = cudaHostGetDevicePointer((void **)&d_idata_r, (void *)h_data_r, 0);
	//assert(!cudaStat);
}

void loadWeights(char * filename){
	sprintf(fn_weight,"%s%s" ,dir,filename);
	printf("Read weights from: %s\n",fn_weight);
	FILE * f_weight;
	f_weight = fopen(fn_weight, "rb");
	size_t size1 = fread(h_weights_r, sizeof(float), nr_rows_A * nr_cols_A *2*N_FBIN, f_weight);
	fclose(f_weight);
}

__global__ void realToComplex(float *idata, cuComplex *odata, int width, int height, int nreps)
{
	  int x = blockDim.x * blockIdx.x + threadIdx.x;
	  int y = blockDim.y * blockIdx.y + threadIdx.y;
	  int z = blockDim.z * blockIdx.z + threadIdx.z;

	  odata[x*height*width + y*width + z].x = idata[2*(x*height*width + y*width + z)];
	  odata[x*height*width + y*width + z].y = idata[2*(x*height*width + y*width + z)+1];
}

__global__ void realDataToComplex(signed char *idata, cuComplex *odata, int width, int height, int nreps)
{
	  int x = blockDim.x * blockIdx.x + threadIdx.x;
	  int y = blockDim.y * blockIdx.y + threadIdx.y;
	  int z = blockDim.z * blockIdx.z + threadIdx.z;

	  //odata[x*height*width + y*width + z].x = idata[2*(x*height*width + y*width + z)];
	  //odata[x*height*width + y*width + z].y = idata[2*(x*height*width + y*width + z)+1];
	  odata[x*height*width + y*width + z].x = (int)idata[2*(x*height*width + y*width + z)]*1.0f;
	  odata[x*height*width + y*width + z].y = (int)idata[2*(x*height*width + y*width + z)+1]*1.0f;
	  //printf("%3.0f %3.0f  ",odata[x*height*width + y*width + z].x,odata[x*height*width + y*width + z].y);
}
__global__ void copyData(cuComplex *idata, cuComplex *odata)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  odata[x*N_ELEM*N_TSAMP + y*N_ELEM + z] = idata[x*N_ELEM*N_TSAMP + y*N_ELEM + z];
}

__global__ void transposeNetData(cuComplex *idata, cuComplex *odata)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  // N_PACK_PER_ELEM, N_TSAMP_PER_PACK, N_ELEM_PER_PACK*N_FBIN
  int in_p = x*N_TSAMP_PER_PACK*N_ELEM_PER_PACK*N_FBIN+y*N_ELEM_PER_PACK*N_FBIN+z;
  int out_p = y*N_PACK_PER_ELEM*N_ELEM_PER_PACK*N_FBIN+x*N_ELEM_PER_PACK*N_FBIN+z;
  odata[out_p] = idata[in_p];
  /*__syncthreads();
  for (int i=0;i<N_PACK_PER_TSAMP;i++){
	  odata[4096*12*i+out_p] = idata[4096*12*i+in_p];
  }
  __syncthreads();*/
}


__global__ void transposeData(cuComplex *idata, cuComplex *odata)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  //odata[z*N_ELEM*N_TSAMP + y*N_ELEM + x] = idata[x*N_ELEM*N_FBIN + y*N_FBIN + z];
  odata[z*N_ELEM*N_TSAMP + x*N_ELEM + y] = idata[x*N_ELEM*N_FBIN + y*N_FBIN + z];
}

__global__ void transposeData2(cuComplex *idata, cuComplex *odata)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  odata[y*N_TSAMP*N_ELEM + x*N_ELEM + z] = idata[x*N_FBIN*N_ELEM + y*N_ELEM + z];
}

__global__ void calcStokes(cuComplex * idata, float * odata) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	int m,n;
	//__syncthreads();
	//Dimension: N_FBIN x N_TSAMP x N_BEAM_STOKES
	m = N_STOKES*(x*N_TSAMP*N_BEAM_STOKES+y*N_BEAM_STOKES+z);
	n = 2*(x*N_TSAMP*N_BEAM_STOKES+y*N_BEAM_STOKES+z);

	// Re(X)^2 + Im(X)^2
	odata[m] = idata[n].x * idata[n].x + idata[n+1].x * idata[n+1].x;
	// Re(Y)^2 + Im(Y)^2
	odata[m+1] = idata[n].y * idata[n].y + idata[n+1].y * idata[n+1].y;
	// Re(XY*)
	odata[m+2] = idata[n].x * idata[n+1].x + idata[n].y * idata[n+1].y;
	// Im(XY*)
	odata[m+3] = idata[n].y * idata[n+1].x - idata[n].x * idata[n+1].y;
	//__syncthreads();
}

__global__ void calcPWR(cuComplex * idata, float * odata) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	int i;
	//__syncthreads();
	//Dimension: N_FBIN x N_TSAMP x N_BEAM_STOKES
	i = x*N_TSAMP*N_BEAM_STOKES+y*N_BEAM_STOKES+z;
	// Power
	odata[i] = idata[i].x * idata[i].x + idata[i].y * idata[i].y;
}


__global__ void transposeStokes(float *idata, float *odata)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  //dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS,N_BEAM_STOKES*N_STOKES/BLOCK_COLS);
  odata[x*N_TSAMP*N_BEAM_STOKES*N_STOKES+z*N_TSAMP+y] = idata[x*N_TSAMP*N_BEAM_STOKES*N_STOKES+y*N_BEAM_STOKES*N_STOKES+z];
}

__global__ void accuStokes(float *idata, float *odata){
	  int x = blockDim.x * blockIdx.x + threadIdx.x;
	  int y = blockDim.y * blockIdx.y + threadIdx.y;
	  int z = blockDim.z * blockIdx.z + threadIdx.z;
	  int m;
	  //__syncthreads();
	  m = x*N_BEAM_STOKES*N_STOKES*N_ACCU + y*N_ACCU + z;
	  for(int i=0;i<N_TSAMP_ACCU;++i){
		  odata[m]+=idata[m*N_TSAMP_ACCU+i];
	  }
	  //__syncthreads();
}

__global__ void accuPWR(float *idata, float *odata){
	  int x = blockDim.x * blockIdx.x + threadIdx.x;
	  int y = blockDim.y * blockIdx.y + threadIdx.y;
	  int z = blockDim.z * blockIdx.z + threadIdx.z;
	  int m;
	  //__syncthreads();
	  m = x*N_ACCU*N_BEAM_STOKES + y*N_BEAM_STOKES + z;
	  for(int i=0;i<N_TSAMP_ACCU;++i){
		  odata[m]+=idata[m*N_TSAMP_ACCU+i];
	  }
	  //__syncthreads();
}

// Function to free resources
void bfCleanup(){
	// Free pinned memory
	cudaFreeHost(h_arr_A);
	cudaFreeHost(h_arr_B);
	cudaFreeHost(h_arr_C);
	cudaFreeHost(h_weights_r);
	//cudaFreeHost(h_data_r);
	cudaFreeHost(h_beamformed);
	//cudaFreeHost(h_accu_stokes);
	// Free GPU memory
	if (d_tdata != NULL) {
		cudaFree(d_tdata);
	}

	if (d_idata != NULL) {
		cudaFree(d_idata);
	}

	if (d_net_data != NULL) {
		cudaFree(d_net_data);
	}

	if (d_weights != NULL) {
		cudaFree(d_weights);
	}

	if (d_beamformed != NULL) {
		cudaFree(d_beamformed);
	}

	if (d_stokes_out != NULL) {
		cudaFree(d_stokes_out);
	}

	if (d_power_out != NULL) {
		cudaFree(d_power_out);
	}

	if (d_accu_stokes_in != NULL) {
		cudaFree(d_accu_stokes_in);
	}
	if (d_accu_stokes != NULL) {
		cudaFree(d_accu_stokes);
	}
	if (d_arr_A != NULL) {
		cudaFree(d_arr_A);
	}

	if (d_arr_B != NULL) {
		cudaFree(d_arr_B);
	}

	if (d_arr_C != NULL) {
		cudaFree(d_arr_C);
	}
	// Free up and release cublas handle
	cublasDestroy(handle);
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}

//int main(int argc, char *argv[])
//void runBeamformer(signed char * data_in, float * data_out)
void runBeamformer()
//void runBeamformer(signed char * data_in)
{
	if (GPU_TEST_PRINT) {begin_main = clock();}

	//print(cudaGetDeviceCount());
	//print(cudaGetDeviceProperties(0));

	//if(argc>1){
	//cuda_core = atoi(argv[1]);}
	//cudaSetDevice(cuda_core);

	//sprintf(dir_output,"%s%s%d%s" ,dir,"gpu",cuda_core,"/");


	// Convert to complex numbers
	if (GPU_TEST_PRINT) {begin = clock();}
	// Weights in dimension of N_FBIN x N_BEAM x N_ELE
	TILE_DIM = 4;
	BLOCK_ROWS = 1;
	BLOCK_COLS = 8;
	dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
	dimGrid = dim3(N_ELEM/TILE_DIM, N_BEAM/BLOCK_ROWS, N_FBIN/BLOCK_COLS);
	//dimBlock = dim3(16, 16, 2);// number of threads per block must be less than 1024
	cudaDeviceSynchronize();
	realToComplex<<<dimGrid,dimBlock>>>(d_weights_r, d_weights, N_FBIN, N_BEAM, N_ELEM);
	cudaDeviceSynchronize();
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("**************************************************************************\n");
		printf("Convert weights to complex numbers elapsed: %3.3f ms\n",time_spent);
	}

	// Read data from file
	//begin = clock();
	//sprintf(fn_data,"%s%s%d%s%d%s%d%s" ,dir,"data_",N_FBIN,"x",N_ELEM,"x",N_TSAMP,".bin");
	/*sprintf(fn_data,"%s%s%d%s%d%s%d%s%d%s" ,dir,"data",cuda_core,"_",N_FBIN,"x",N_ELEM,"x",N_TSAMP,".bin");
	FILE * f_data;
	f_data = fopen(fn_data, "rb");
	size_t size2 = fread(h_data_r, sizeof(float), 2*nr_rows_B * nr_cols_B *N_FBIN, f_data);
	fclose(f_data);*/
	//h_data_r = data_in;
	//cudaStat = cudaHostGetDevicePointer((void **)&d_idata_r, (void *)data_in, 0);
	//assert(!cudaStat);
	//memcpy(h_data_r,data_in,N_INPUTS*2*sizeof(signed char));
	//cudaMemcpy(d_idata_r,h_data_r,2*nr_rows_B * nr_cols_B *N_FBIN*sizeof(float),cudaMemcpyHostToDevice);
	/*for (int i=0;i<8192;i++){
		printf("%d ",h_data_r[i]);
	}*/
	/*if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Read data from %s elapsed: %3.3f ms\n",fn_data,time_spent);
	}*/
	// Convert to complex numbers
	if (GPU_TEST_PRINT) {begin = clock();}
	cudaDeviceSynchronize();
	// If input data dimension is: N_TSAMP x N_ELE x N_FBIN
	if(TRANS_INPUT==1){
		//printf("TRANS_INPUT is: %d\n",TRANS_INPUT);
		//dimGrid = dim3(N_TSAMP/TILE_DIM, N_ELEM/BLOCK_ROWS,N_FBIN/BLOCK_COLS);
		TILE_DIM = 32;
		BLOCK_ROWS = 8;
		BLOCK_COLS = 4;
		dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
		dimGrid = dim3(N_TSAMP/TILE_DIM, N_FBIN/BLOCK_ROWS,N_ELEM/BLOCK_COLS);
		realDataToComplex<<<dimGrid,dimBlock>>>(d_idata_r, d_idata, N_ELEM, N_FBIN, N_TSAMP);
	}
	else{
		dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS,N_ELEM/BLOCK_COLS);
		realDataToComplex<<<dimGrid,dimBlock>>>(d_idata_r, d_idata, N_ELEM, N_TSAMP, N_FBIN);
	}
	cudaDeviceSynchronize();
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Convert data to complex numbers elapsed: %3.3f ms\n",time_spent);
	}

	if (GPU_TEST_PRINT) {begin = clock();}
	cudaDeviceSynchronize();
	// If transpose input data is needed, then transpose data to dimension: N_FBIN x N_TSAMP x N_ELE
	if(TRANS_INPUT==1){
		if(FAKED_INPUT==1){
			TILE_DIM = 12;
			dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);

			dimGrid = dim3(N_PACK_PER_ELEM/TILE_DIM, N_TSAMP_PER_PACK/BLOCK_ROWS,N_ELEM_PER_PACK*N_FBIN/BLOCK_COLS);
			transposeNetData<<<dimGrid, dimBlock>>>(d_idata, d_net_data);
			for (int i=0;i<N_PACK_PER_TSAMP;i++){
				transposeNetData<<<dimGrid, dimBlock>>>(d_idata+4096*12*i, d_net_data+4096*12*i);
			}
			TILE_DIM = 16;
			dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
			dimGrid = dim3(N_TSAMP/TILE_DIM, N_ELEM/BLOCK_ROWS,N_FBIN/BLOCK_COLS);
			transposeData<<<dimGrid, dimBlock>>>(d_net_data, d_tdata);
		}
		else{
			if(N_POLS==1){
				//printf("N_POLS is: %d\n",N_POLS);
				TILE_DIM = 32;
				BLOCK_ROWS = 8;
				BLOCK_COLS = 4;
				dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
				dimGrid = dim3(N_TSAMP/TILE_DIM, N_FBIN/BLOCK_ROWS,N_ELEM/BLOCK_COLS);
				transposeData2<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
			}
			else{
				dimGrid = dim3(N_TSAMP/TILE_DIM, N_ELEM/BLOCK_ROWS,N_FBIN/BLOCK_COLS);
				transposeData<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
			}
		}
	}
	else{
		dimGrid = dim3(N_TSAMP/TILE_DIM, N_ELEM/BLOCK_ROWS,N_FBIN/BLOCK_COLS);
		copyData<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
	}
	/*if(TRANS_INPUT==1){
		transposeData<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
	}
	else{
		copyData<<<dimGrid, dimBlock>>>(d_idata, d_tdata);
	}*/
	cudaDeviceSynchronize();
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Transpose data elapsed: %3.3f ms\n",time_spent);
	}

	// Execute matrix multipulication kernel
	if (GPU_TEST_PRINT) {cudaEventRecord( start, 0 ) ;}
	// Leading dimensions are always the rows of each matrix since the data is stored in a column-wise order.
	int lda=nr_rows_A, ldb=nr_rows_B, ldc=nr_rows_C;
	cuComplex alf;
	cuComplex bet;
	alf.x = 1;
	alf.y = 0;
	bet.x = 0;
	bet.y = 0;
	int batchCount = N_FBIN; 				// There must be the same number of batches in each array.
	cudaDeviceSynchronize();
	stat = cublasCgemmBatched(
			handle,							// handle to the cuBLAS library context.
			CUBLAS_OP_N,					// Operation on matrices within array A.
			CUBLAS_OP_N,					// Operation on matrices within array B.
			nr_rows_A,						// Number of rows in matrix A and C.
			nr_cols_B,						// Number of columns in matrix B and C.
			nr_cols_A,						// Number of columns and rows in matrix A and B respectively.
			&alf,							// Scalar used for multiplication.
			(const cuComplex **)d_arr_A,	// Weight array of pointers.
			lda,							// Leading dimension of each batch or matrix in array A.
			(const cuComplex **)d_arr_B,	// Data array of pointers.
			ldb,							// Leading dimension of each batch or matrix in array B.
			&bet,							// Scalar used for multiplication.
			(cuComplex **)d_arr_C,			// Output array of pointers.
			ldc,							// Leading dimension of each batch or matrix in array C.
			batchCount);					// Number of batches in each array.

	cudaDeviceSynchronize();
	if (stat == CUBLAS_STATUS_INVALID_VALUE) {
		printf("RTBF: Invalid CUBLAS values\n");
	} else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
		printf("RTBF: Execution failed.\n");
	}

	if(stat != CUBLAS_STATUS_SUCCESS){
		cerr << "cublasCgemmBatched failed" << endl;
		exit(1);
	}
	assert(!cudaGetLastError());
	if (GPU_TEST_PRINT) {
		cudaEventRecord( stop, 0 ) ;
		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &elapsedTime,start, stop );
		printf( "Matrix multiplication kernel(cublasSgemmBatched) duration: %3.3f ms\n", elapsedTime );
	}
	if (RECORD_BF_RAW==1){
		// copy beamformed data back to host, zero copy cannot map memory from GPU to CPU
		if (GPU_TEST_PRINT) {begin = clock();}
		cudaDeviceSynchronize();
		cudaStat = cudaMemcpy(h_beamformed, d_beamformed, N_OUTPUTS_BF*sizeof(cuComplex), cudaMemcpyDeviceToHost);
		assert(!cudaStat);
		//cudaStat = cudaHostGetDevicePointer((void **)&d_beamformed, (void *)h_beamformed, 0);
		//assert(!cudaStat);
		cudaDeviceSynchronize();
		//dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS, N_BEAM/BLOCK_COLS);
		//copyData<<<dimGrid, dimBlock>>>(d_beamformed, h_beamformed);
		if (GPU_TEST_PRINT) {
			end = clock();
			time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
			printf("Copy beamformed data back to host elapsed: %3.3f ms\n",time_spent);
		}
		if (GPU_TEST_PRINT) {begin = clock();}
		// Write beamformed result to file
		sprintf(fn_output_bf,"%s%s%d%s%d%s%d%s" ,dir_output,"output_bf_",N_FBIN,"x",N_BEAM,"x",N_TSAMP,".bin");///home/peix/workspace/paf_sim/output_**.bin
		FILE * f_output_bf;
		f_output_bf = fopen(fn_output_bf, "wb");
		fwrite(h_beamformed, sizeof(cuComplex), N_OUTPUTS_BF, f_output_bf);
		fclose(f_output_bf);
		if (GPU_TEST_PRINT) {
			end = clock();
			time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
			printf("Write beamformed result to file elapsed: %3.3f ms\n",time_spent);
		}
	}
	if (GPU_TEST_PRINT) {begin = clock();}
	cudaDeviceSynchronize();
	if (N_STOKES==4){
		dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS, N_BEAM_STOKES/BLOCK_COLS);
		calcStokes<<<dimGrid,dimBlock>>>(d_beamformed, d_stokes_out);
		dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS, N_BEAM_STOKES*N_STOKES/BLOCK_COLS);
		transposeStokes<<<dimGrid,dimBlock>>>(d_stokes_out,d_accu_stokes_in);
		cudaMemset(d_accu_stokes,0,N_OUTPUTS*sizeof(float));
		dimGrid = dim3(N_FBIN/TILE_DIM, N_BEAM_STOKES*N_STOKES/BLOCK_ROWS, N_ACCU/BLOCK_COLS);
		accuStokes<<<dimGrid,dimBlock>>>(d_accu_stokes_in, d_accu_stokes);}
	else if (N_STOKES==1){
		TILE_DIM = 8;
		BLOCK_ROWS = 128;
		BLOCK_COLS = 1;
		dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
		dimGrid = dim3(N_FBIN/TILE_DIM, N_TSAMP/BLOCK_ROWS, N_BEAM_STOKES/BLOCK_COLS);
		calcPWR<<<dimGrid,dimBlock>>>(d_beamformed, d_power_out);
		cudaMemset(d_accu_stokes,0,N_OUTPUTS*sizeof(float));
		TILE_DIM = 8;
		BLOCK_ROWS = 1;
		BLOCK_COLS = 1;
		dimBlock = dim3(TILE_DIM,BLOCK_ROWS,BLOCK_COLS);
		//dimGrid = dim3(N_FBIN/TILE_DIM, N_BEAM_STOKES*N_STOKES/BLOCK_ROWS, N_ACCU/BLOCK_COLS);
		dimGrid = dim3(N_FBIN/TILE_DIM, N_ACCU/BLOCK_ROWS, N_BEAM_STOKES*N_STOKES/BLOCK_COLS);
		accuPWR<<<dimGrid,dimBlock>>>(d_power_out, d_accu_stokes);
	}
	cudaDeviceSynchronize();
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Calculate Stokes parameter and accumulate data elapsed: %3.3f ms\n",time_spent);
	}

	begin = clock();
	cudaDeviceSynchronize();
	// copy accumulated Stokes data back to host
	cudaStat = cudaMemcpy(h_accu_stokes, d_accu_stokes, N_OUTPUTS*sizeof(float), cudaMemcpyDeviceToHost);
	assert(!cudaStat);
	//dimGrid = dim3(N_FBIN/TILE_DIM, N_BEAM_STOKES*N_STOKES/BLOCK_ROWS, N_ACCU/BLOCK_COLS);
	//copyDataReal<<<dimGrid, dimBlock>>>(d_accu_stokes, h_accu_stokes);
	cudaDeviceSynchronize();
	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Copy accumulated Stokes data to host elapsed: %3.3f ms\n",time_spent);
	}
	//begin = clock();
	/*sprintf(fn_output,"%s%s%d%s%d%s%d%s" ,dir_output,"output_",N_FBIN,"x",N_BEAM_STOKES,"x",N_ACCU,".bin");///home/peix/workspace/paf_sim/output_**.bin
	//sprintf(fn_output,"%s%s%d%s%d%s%d%s" ,dir_output,"output_",N_FBIN,"x",N_BEAM_STOKES,"x",N_ACCU,".bin");///home/peix/workspace/paf_sim/output_**.bin
	FILE * f_output;
	f_output = fopen(fn_output, "wb");
	fwrite(h_accu_stokes, sizeof(float), N_OUTPUTS, f_output);
	fclose(f_output);*/
	//data_out = h_accu_stokes;
	//memcpy(data_out,h_accu_stokes,N_OUTPUTS*sizeof(float));
    /*for (int i=0;i<N_FBIN;i++){
    	for (int j=0;j<N_BEAM_STOKES*N_STOKES;j++){
    		if(j<4){
    			printf("%3.2f ",h_accu_stokes[i*N_BEAM_STOKES*N_STOKES*N_ACCU+j*N_ACCU]);
    		}
    	}
    }*/
	/*if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Write accumulated Stokes data to file elapsed: %3.3f ms\n",time_spent);
	}*/
	/*begin = clock();
	// Free resources
	//bfCleanup();

	if (GPU_TEST_PRINT) {
		end = clock();
		time_spent = (double)(end - begin)/CLOCKS_PER_SEC*1000;
		printf("Free memory elapsed: %3.3f ms\n",time_spent);
	}*/
	if (GPU_TEST_PRINT) {
		end_main = clock();
		time_spent = (double)(end_main - begin_main)/CLOCKS_PER_SEC*1000;
		printf("The run_beamformer program totally elapsed: %3.3f ms\n",time_spent);
		//printf("**************************************************************************\n");
	}
    //return 0;
}
