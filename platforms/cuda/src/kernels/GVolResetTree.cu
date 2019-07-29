
#define PI (3.14159265359f)

/**
 * Initialize tree for execution, set Processed to 0, OKtoProcess=1 for leaves and out-of-bound,
 * reset self volume accumulators.
 */

__device__ void resetTreeCounters(unsigned const int padded_tree_size,
                                  unsigned const int tree_size,
                                  unsigned const int offset,
                                  int *restrict ovProcessedFlag,
                                  int *restrict ovOKtoProcessFlag,
                                  const int *restrict ovChildrenStartIndex,
                                  const int *restrict ovChildrenCount,
                                  int *restrict ovChildrenReported) {
    const unsigned int id = threadIdx.x;  //the index of this thread in the workgroup
    const unsigned int nblock = blockDim.x; //size of work group
    unsigned int begin = offset + id;
    unsigned int size = offset + tree_size;
    unsigned int end = offset + padded_tree_size;

    for (int slot = begin; slot < end; slot += nblock) {
        ovProcessedFlag[slot] = (slot >= size) ? 1 : 0; //mark slots with overlaps as not processed
    }
    for (int slot = begin; slot < end; slot += nblock) {
        ovOKtoProcessFlag[slot] = (slot >= size) ? 0 : (ovChildrenCount[slot] == 0 ? 1
                                                                                   : 0); //marks leaf nodes (no children) as ok to process
    }
    for (int slot = begin; slot < end; slot += nblock) {
        ovChildrenReported[slot] = 0;
    }
}


//assume num. groups = num. tree sections
__global__ void resetSelfVolumes(const int ntrees,
                                 const int *restrict ovTreePointer,
                                 const int *restrict ovAtomTreePointer,
                                 const int *restrict ovAtomTreeSize,
                                 const int *restrict ovAtomTreePaddedSize,
                                 const int *restrict ovChildrenStartIndex,
                                 const int *restrict ovChildrenCount,
                                 int *restrict ovProcessedFlag,
                                 int *restrict ovOKtoProcessFlag,
                                 int *restrict ovChildrenReported,
                                 int *restrict PanicButton) {
    unsigned int tree = blockIdx.x;      //initial tree
    if (PanicButton[0] > 0) return;
    while (tree < ntrees) {

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
__device__ void resetTreeSection(unsigned const int padded_tree_size,
                                 unsigned const int offset,
                                 int *restrict ovLevel,
                                 real *restrict ovVolume,
                                 real *restrict ovVsp,
                                 real *restrict ovVSfp,
                                 real *restrict ovSelfVolume,
                                 real *restrict ovVolEnergy,
                                 int *restrict ovLastAtom,
                                 int *restrict ovRootIndex,
                                 int *restrict ovChildrenStartIndex,
                                 int *restrict ovChildrenCount,
                                 real4 *restrict ovDV1,
                                 real4 *restrict ovDV2,
                                 int *restrict ovProcessedFlag,
                                 int *restrict ovOKtoProcessFlag,
                                 int *restrict ovChildrenReported) {
    const unsigned int nblock = blockDim.x; //size of thread block
    const unsigned int id = threadIdx.x;  //the index of this thread in the warp

    unsigned int begin = offset + id;
    unsigned int end = offset + padded_tree_size;

    for (int slot = begin; slot < end; slot += nblock) ovLevel[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovVsp[slot] = 1;
    for (int slot = begin; slot < end; slot += nblock) ovVSfp[slot] = 1;
    for (int slot = begin; slot < end; slot += nblock) ovSelfVolume[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovVolEnergy[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovLastAtom[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovRootIndex[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenStartIndex[slot] = -1;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenCount[slot] = 0;
    //for(int slot=begin; slot<end ; slot+=nblock) ovDV1[slot] = (real4)0;
    //for(int slot=begin; slot<end ; slot+=nblock) ovDV2[slot] = (real4)0;
    for (int slot = begin; slot < end; slot += nblock) ovDV1[slot] = make_real4(0);
    for (int slot = begin; slot < end; slot += nblock) ovDV2[slot] = make_real4(0);
    for (int slot = begin; slot < end; slot += nblock) ovProcessedFlag[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovOKtoProcessFlag[slot] = 0;
    for (int slot = begin; slot < end; slot += nblock) ovChildrenReported[slot] = 0;
}


__global__ void resetBuffer(unsigned const int bufferSize,
                            unsigned const int numBuffers,
                            real4 *restrict ovAtomBuffer,
                            real *restrict selfVolumeBuffer,
                            long *restrict selfVolumeBuffer_long,
                            long *restrict gradBuffers_long) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
//#ifdef SUPPORTS_64_BIT_ATOMICS
    while (id < bufferSize) {
        selfVolumeBuffer_long[id] = 0;
        gradBuffers_long[id] = 0;
        gradBuffers_long[id + bufferSize] = 0;
        gradBuffers_long[id + 2 * bufferSize] = 0;
        gradBuffers_long[id + 3 * bufferSize] = 0;
        id += blockDim.x * gridDim.x;
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
                          const int *restrict ovTreePointer,
                          const int *restrict ovAtomTreePointer,
                          int *restrict ovAtomTreeSize,
                          const int *restrict ovAtomTreePaddedSize,
                          int *restrict ovLevel,
                          real *restrict ovVolume,
                          real *restrict ovVsp,
                          real *restrict ovVSfp,
                          real *restrict ovSelfVolume,
                          real *restrict ovVolEnergy,
                          int *restrict ovLastAtom,
                          int *restrict ovRootIndex,
                          int *restrict ovChildrenStartIndex,
                          int *restrict ovChildrenCount,
                          real4 *restrict ovDV1,
                          real4 *restrict ovDV2,
                          int *restrict ovProcessedFlag,
                          int *restrict ovOKtoProcessFlag,
                          int *restrict ovChildrenReported,
                          int *restrict ovAtomTreeLock,
                          int *restrict NIterations) {

    unsigned int section = blockIdx.x; // initial assignment of warp to tree section
    while (section < ntrees) {
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
        if (threadIdx.x == 0) {
            ovAtomTreeLock[section] = 0;
            NIterations[section] = 0;
        }
        section += gridDim.x; //next section
    }
}
