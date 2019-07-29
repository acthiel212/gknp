#define PI (3.14159265359f)


/**
 * Reduce the atom self volumes
 * energyBuffer could be linked to the selfVolumeBuffer, depending on the use
 */

__global__ void reduceSelfVolumes_buffer(int num_atoms, int padded_num_atoms,
				       int numBuffers, 
				       __device__ const int*   restrict ovAtomTreePointer,
				       __device__ const real4* restrict ovAtomBuffer,
//#ifdef SUPPORTS_64_BIT_ATOMICS
				       __device__       long*  restrict gradBuffers_long,
				       __device__       long*  restrict selfVolumeBuffer_long,
//#endif
				       __device__       real*  restrict selfVolumeBuffer,
				       __device__       real*  restrict selfVolume,
				       __device__ const real* restrict global_gaussian_volume, //atomic Gaussian volume
				       __device__ const real* restrict global_atomic_gamma, //atomic gammas
				       __device__       real4* restrict grad //gradient wrt to atom positions and volume
){
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  int totalSize = padded_num_atoms*numBuffers;
//#ifdef SUPPORTS_64_BIT_ATOMICS
  real scale = 1/(real) 0x100000000;
//#endif
  
  //accumulate self volumes
  unsigned int atom = id;
  while (atom < num_atoms) {
//#ifdef SUPPORTS_64_BIT_ATOMICS
    // copy self volumes and gradients from long energy buffer to regular one
    selfVolume[atom] = scale*selfVolumeBuffer_long[atom];
    grad[atom].x = scale*gradBuffers_long[atom];
    grad[atom].y = scale*gradBuffers_long[atom+padded_num_atoms];
    grad[atom].z = scale*gradBuffers_long[atom+2*padded_num_atoms];
    // divide gradient with respect to volume by volume of atom
    if(global_gaussian_volume[atom] > 0){
      grad[atom].w = scale*gradBuffers_long[atom+3*padded_num_atoms]/global_gaussian_volume[atom];
    }else{
      grad[atom].w = 0;
    }
//#else
//    real sum = 0;
//    for (int i = atom; i < totalSize; i += padded_num_atoms) sum += selfVolumeBuffer[i];
//    selfVolume[atom] = sum;
//    //real4 sum4 = 0;
//    real4 sum4 = make_real4(0);
//    //for (int i = atom; i < totalSize; i += padded_num_atoms) sum4 += ovAtomBuffer[i];
//    //grad[atom].xyz = sum4.xyz;
//    for (int i = atom; i < totalSize; i += padded_num_atoms) sum4 = make_real4(sum4.x+ovAtomBuffer[i].x,
//                                                                               sum4.y+ovAtomBuffer[i].y,
//                                                                               sum4.z+ovAtomBuffer[i].z,
//                                                                               sum4.w+ovAtomBuffer[i].w);
//    grad[atom] = make_real4(sum4.x, sum4.y, sum4.z, grad[atom].w);
//    if(global_gaussian_volume[atom] > 0){
//      grad[atom].w = sum4.w/global_gaussian_volume[atom];
//    }else{
//      grad[atom].w = 0.;
//    }
//#endif
   atom += blockDim.x*gridDim.x;
 }
  //TODOLater: Global memory fence needed or syncthreads sufficient?
 __syncthreads();

 /*
 if(update_energy > 0){
   unsigned int atom = id;
   while (atom < NUM_ATOMS_TREE) {
     // volume energy is stored at the 1-body level
     unsigned int slot = ovAtomTreePointer[atom];
     energyBuffer[id] += ovVolEnergy[slot];
     //alternative to the above, should give the same answer
     //energyBuffer[atom] += global_atomic_gamma[atom]*selfVolume[atom];
 
#ifdef SUPPORTS_64_BIT_ATOMICS
     // do nothing with forces, they are stored in computeSelfVolumes kernel
     // divide gradient with respect to volume by volume of atom
#else
     real4 sum = 0;
     for (int i = atom; i < totalSize; i += padded_num_atoms) sum += ovAtomBuffer[i];
     forceBuffers[atom].xyz -= sum.xyz;
#endif
     atom += blockDim.x*gridDim.x;
   }
   __syncthreads(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
 }
 */
 
}


__global__ void updateSelfVolumesForces(int update_energy,
				      int num_atoms,
				      int padded_num_atoms,
				      __device__ const int*  restrict ovAtomTreePointer,
				      __device__ const real* restrict ovVolEnergy,
				      __device__ const real4* restrict grad, //gradient wrt to atom positions and volume
//#ifdef SUPPORTS_64_BIT_ATOMICS
				      __device__ long*   restrict forceBuffers,
//#else
//				      __device__ real4* restrict forceBuffers,
//#endif
				      __device__ mixed*  restrict energyBuffer
){
  //update OpenMM's energies and forces
  unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  unsigned int atom = id;
  while (atom < num_atoms) {
    // volume energy is stored at the 1-body level
    if(update_energy > 0){
      unsigned int slot = ovAtomTreePointer[atom];
      energyBuffer[id] += ovVolEnergy[slot];
      //alternative to the above, should give the same answer
      //energyBuffer[atom] += wen*global_atomic_gamma[atom]*selfVolume[atom];
    }
//#ifdef SUPPORTS_64_BIT_ATOMICS
//    atom_add(&forceBuffers[atom                     ], (long)(-grad[atom].x*0x100000000));
//    atom_add(&forceBuffers[atom +   padded_num_atoms], (long)(-grad[atom].y*0x100000000));
//    atom_add(&forceBuffers[atom + 2*padded_num_atoms], (long)(-grad[atom].z*0x100000000));
      atomicAdd(&forceBuffers[atom                     ], (long)(-grad[atom].x*0x100000000));
      atomicAdd(&forceBuffers[atom +   padded_num_atoms], (long)(-grad[atom].y*0x100000000));
      atomicAdd(&forceBuffers[atom + 2*padded_num_atoms], (long)(-grad[atom].z*0x100000000));
//#else
//    //forceBuffers[atom].xyz -= grad[atom].xyz;
//    forceBuffers[atom].xyz = make_real4(forceBuffers[atom].x - grad[atom].x,
//                                        forceBuffers[atom].y - grad[atom].y,
//                                        forceBuffers[atom].z - grad[atom].z,
//                                        forceBuffers[atom].w);
//
//#endif
    atom += blockDim.x*gridDim.x;
 }
  //TODOLater: Global memory fence needed or syncthreads sufficient?
 __syncthreads();
 
}
