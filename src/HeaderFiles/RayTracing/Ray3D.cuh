#ifndef RAY3D_CUH
#define RAY3D_CUH

#include <cuda_runtime.h>

struct Ray3D {
    float directionX;
    float directionY;
    float directionZ;
    float originX;
    float originY;
    float originZ;

    __device__ Ray3D(const float& directionX,const float& directionY,const float& directionZ,const float& originX,const float& originY,const float& originZ) :
    directionX(directionX),directionY(directionY),directionZ(directionZ),originX(originX),originY(originY),originZ(originZ) {
        
    }

    __device__ inline void normalize() {
        float inverseLength = 1.0/sqrtf(directionX*directionX + directionY*directionY + directionZ*directionZ);
        this->directionX *= inverseLength;
        this->directionY *= inverseLength;
        this->directionZ *= inverseLength;
    }

};


#endif