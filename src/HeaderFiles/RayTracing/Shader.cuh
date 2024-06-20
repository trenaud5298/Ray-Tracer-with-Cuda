#ifndef SHADER_CUH
#define SHADER_CUH

#include "../World/Material.h"
#include "./Ray3D.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "../../../lib/SDL/include/SDL.h"
#include <chrono>

class Shader {

private:
    static float* GPU_RAW_IMAGE_DATA;
    static unsigned char* GPU_IMAGE_DATA;

    //CPU IMAGE DATA
    static unsigned char* hostImageData;
   
    //Camera Data


    //Thread Info
    static dim3 threadsPerBlock;
    static dim3 blocksPerGrid; 

    //Frame Info
    static int currentFrame;

    //Shader Allocation of GPU Memory
    static void allocateMemory();
    static void freeDynamicMemory();


public:
    static void initShader(const int& imageWidth, const int& imageHeight);
    static void renderFrame();
    static void retrieveImageData();
    static void saveImage(const std::string& fileName);
    static void updateImageSize(const int& newImageWidth, const int& newImageHeight);
    static void updateCameraData(float* cameraData);
    static void updateWorldData(float*& newWorldData,size_t& sizeOfWorldData);

};


#endif