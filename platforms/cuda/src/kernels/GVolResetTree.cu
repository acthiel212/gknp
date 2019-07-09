//TODO: Cuda Analog or is it even necessary?
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#ifdef SUPPORTS_64_BIT_ATOMICS
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
//#endif

#define PI (3.14159265359f)

/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */

__device__ void resetTreeCounters(
  unsigned          const int            padded_tree_size,
  unsigned          const int            tree_size,
  unsigned          const int            offset,
  __device__                int*   restrict ovProcessedFlag,
  __device__                int*   restrict ovOKtoProcessFlag,
  __device__          const int*   restrict ovChildrenStartIndex,
  __device__          const int*   restrict ovChildrenCount,
  __device__                int*   restrict ovChildrenReported
){
  const unsigned int id = threadIdx.x;  //the index of this thread in the workgroup
  const unsigned int nblock = blockDim.x; //size of work group
  unsigned int begin = offset + id;
  unsigned int size = offset + tree_size;
  unsigned int end  = offset + padded_tree_size;

  for(int slot=begin; slot<end ; slot+=nblock){
    ovProcessedFlag[slot] = (slot >= size) ? 1 : 0; //mark slots with overlaps as not processed
  }
  for(int slot=begin; slot<end ; slot+=nblock){
    ovOKtoProcessFlag[slot] = (slot >= size) ? 0 : 
      ( ovChildrenCount[slot] == 0 ? 1 : 0); //marks leaf nodes (no children) as ok to process
  }
  for(int slot=begin; slot<end ; slot+=nblock){
    ovChildrenReported[slot] = 0;
  }
}


//assume num. groups = num. tree sections
__global__ void resetSelfVolumes(const int ntrees,
			       __device__ const int*   restrict ovTreePointer,
			       __device__ const int*   restrict ovAtomTreePointer,
			       __device__ const int*   restrict ovAtomTreeSize,
			       __device__ const int*   restrict ovAtomTreePaddedSize,
			       __device__ const int*   restrict ovChildrenStartIndex,
			       __device__ const int*   restrict ovChildrenCount,
			       __device__       int*   restrict ovProcessedFlag,
			       __device__       int*   restrict ovOKtoProcessFlag,
			       __device__       int*   restrict ovChildrenReported,
			       __device__       int*   restrict PanicButton
){
    unsigned int tree = blockIdx.x;      //initial tree
    if(PanicButton[0] > 0) return;
    while (tree < ntrees){

      unsigned int offset = ovTreePointer[tree];
      unsigned int tree_size = ovAtomTreeSize[tree];
      unsigned int padded_tree_size = ovAtomTreePaddedSize[tree];
      resetTreeCounters(padded_tree_size, tree_size, offset, 
			ovProcessedFlag,
			ovOKtoProcessFlag,
			ovChildrenStartIndex,
			ovChildrenCount,
			ovChildrenReported);
      tree += gridDim.x;
    }
}


/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */
__device__ void resetTreeSection(
		      unsigned const int padded_tree_size,
		      unsigned const int offset,
		      __device__       int*   restrict ovLevel,
		      __device__       real*  restrict ovVolume,
		      __device__       real*  restrict ovVsp,
		      __device__       real*  restrict ovVSfp,
		      __device__       real*  restrict ovSelfVolume,
		      __device__       real*  restrict ovVolEnergy,
		      __device__       int*   restrict ovLastAtom,
		      __device__       int*   restrict ovRootIndex,
		      __device__       int*   restrict ovChildrenStartIndex,
		      __device__       int*   restrict ovChildrenCount,
		      __device__       real4* restrict ovDV1,
		      __device__       real4* restrict ovDV2,
		      __device__       int*   restrict ovProcessedFlag,
		      __device__       int*   restrict ovOKtoProcessFlag,
		      __device__       int*   restrict ovChildrenReported){
  const unsigned int nblock = blockDim.x; //size of thread block
  const unsigned int id = threadIdx.x;  //the index of this thread in the warp

  unsigned int begin = offset + id;
  unsigned int end  = offset + padded_tree_size;

  for(int slot=begin; slot<end ; slot+=nblock) ovLevel[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovVsp[slot] = 1;
  for(int slot=begin; slot<end ; slot+=nblock) ovVSfp[slot] = 1;
  for(int slot=begin; slot<end ; slot+=nblock) ovSelfVolume[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovVolEnergy[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovLastAtom[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovRootIndex[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenStartIndex[slot] = -1;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenCount[slot] = 0;
  //for(int slot=begin; slot<end ; slot+=nblock) ovDV1[slot] = (real4)0;
  //for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = (real4)0;
  for(int slot=begin; slot<end ; slot+=nblock) ovDV1[slot] = make_real4(0);
  for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = make_real4(0);
  for(int slot=begin; slot<end ; slot+=nblock) ovProcessedFlag[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovOKtoProcessFlag[slot] = 0;
  for(int slot=begin; slot<end ; slot+=nblock) ovChildrenReported[slot] = 0;
}


__global__ void resetBuffer(unsigned const int             bufferSize,
			  unsigned const int             numBuffers,
			  __device__       real4* restrict ovAtomBuffer,
			  __device__        real* restrict selfVolumeBuffer
//#ifdef SUPPORTS_64_BIT_ATOMICS
			  ,
			  __device__ long*   restrict selfVolumeBuffer_long,
			  __device__ long*   restrict gradBuffers_long

//#endif
){
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
//#ifdef SUPPORTS_64_BIT_ATOMICS
  while (id < bufferSize){
    selfVolumeBuffer_long[id] = 0;
    gradBuffers_long[id             ] = 0;
    gradBuffers_long[id+  bufferSize] = 0;
    gradBuffers_long[id+2*bufferSize] = 0;
    gradBuffers_long[id+3*bufferSize] = 0;
    id += blockDim.x*gridDim.x;
  }
//#else
//  while(id < bufferSize*numBuffers){
//    //ovAtomBuffer[id] = (real4)0;
//    ovAtomBuffer[id] = make_real4(0);
//    selfVolumeBuffer[id] = 0;
//    id += blockDim.x*gridDim.x;
//  }
//#endif
//TODOLater: Global memory fence needed or syncthreads sufficient?
  __syncthreads();
}


__global__ void resetTree(const int ntrees,
			__device__ const int*   restrict ovTreePointer,
			__device__ const int*   restrict ovAtomTreePointer,
			__device__       int*   restrict ovAtomTreeSize,
			__device__ const int*   restrict ovAtomTreePaddedSize,
			__device__       int*   restrict ovLevel,
			__device__       real*  restrict ovVolume,
			__device__       real*  restrict ovVsp,
			__device__       real*  restrict ovVSfp,
			__device__       real*  restrict ovSelfVolume,
			__device__       real*  restrict ovVolEnergy,
			__device__       int*   restrict ovLastAtom,
			__device__       int*   restrict ovRootIndex,
			__device__       int*   restrict ovChildrenStartIndex,
			__device__       int*   restrict ovChildrenCount,
			__device__       real4* restrict ovDV1,
			__device__       real4* restrict ovDV2,

			__device__       int*  restrict ovProcessedFlag,
			__device__       int*  restrict ovOKtoProcessFlag,
			__device__       int*  restrict ovChildrenReported,
			__device__       int*  restrict ovAtomTreeLock,
			__device__       int*  restrict NIterations
			
			){


  unsigned int section = blockIdx.x; // initial assignment of warp to tree section
  while(section < ntrees){
    unsigned int offset = ovTreePointer[section];
    unsigned int padded_tree_size = ovAtomTreePaddedSize[section];

    //each block resets one section of the tree
    resetTreeSection(padded_tree_size, offset, 
		     ovLevel,
		     ovVolume,
		     ovVsp,
		     ovVSfp,
		     ovSelfVolume,
		     ovVolEnergy,
		     ovLastAtom,
		     ovRootIndex,
		     ovChildrenStartIndex,
		     ovChildrenCount,
		     ovDV1,
		     ovDV2,
		     ovProcessedFlag,
		     ovOKtoProcessFlag,
		     ovChildrenReported
		     );
    if(threadIdx.x == 0){
      ovAtomTreeLock[section] = 0;
      NIterations[section] = 0;
    }
    section += gridDim.x; //next section
  }
}
