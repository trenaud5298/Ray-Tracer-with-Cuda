#include "../../HeaderFiles/RayTracing/Shader.cuh"

//Constant Memory Decalaration
__constant__ float GPU_WORLD_DATA[15000];
// __constant__ Material GPU_MATERIALS[40]; 

__constant__ size_t GPU_WORLD_SIZE;
__constant__ int GPU_IMAGE_WIDTH;
__constant__ int GPU_IMAGE_HEIGHT;

//Camera Data:
__constant__ float GPU_COSX;
__constant__ float GPU_COSY;
__constant__ float GPU_SINX;
__constant__ float GPU_SINY;
__constant__ float GPU_CAMERA_POS_X;
__constant__ float GPU_CAMERA_POS_Y;
__constant__ float GPU_CAMERA_POS_Z;
__constant__ float GPU_FOV_SCALAR;
__constant__ float GPU_FOCAL_DISTANCE;
__constant__ float GPU_DEPTH_OF_FIELD_STRENGTH;


//Other Static Member Definitions
float* Shader::GPU_RAW_IMAGE_DATA = nullptr;
unsigned char* Shader::GPU_IMAGE_DATA = nullptr;
unsigned char* Shader::hostImageData = nullptr;
dim3 Shader::threadsPerBlock;
dim3 Shader::blocksPerGrid;
int Shader::currentFrame;
cudaStream_t GPU_STREAMS[10];



__device__ Ray3D getInitialRay(const int& x, const int& y) {
    float tempX,tempY,tempZ,offsetX,offsetY,inverseLength;
    offsetX = 0.0f;
    offsetY = 0.0f;
    tempX = (2*x-GPU_IMAGE_WIDTH)/static_cast<float>(GPU_IMAGE_WIDTH) - offsetX;
    tempY = (GPU_IMAGE_HEIGHT-2*y)/static_cast<float>(GPU_IMAGE_HEIGHT) - offsetY;
    tempZ = 1.0f;
    inverseLength = 1.0f/sqrtf(tempX*tempX + tempY*tempY + tempZ*tempZ);
    tempX *= inverseLength;
    tempY *= inverseLength;
    tempZ *= inverseLength;
    inverseLength/*Representing a vertically rotated z coord*/ = tempX * GPU_SINY + tempZ * GPU_COSY;
    return Ray3D(tempX*GPU_COSX + inverseLength*GPU_SINX,tempY*GPU_COSY-tempZ*GPU_SINY,inverseLength*GPU_COSX-tempX*GPU_SINX,GPU_CAMERA_POS_X+offsetX,GPU_CAMERA_POS_Y+offsetY,GPU_CAMERA_POS_Z);
}

__global__ void mainShader(float* GPU_RAW_IMAGE_DATA, unsigned char* GPU_IMAGE_DATA) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < GPU_IMAGE_WIDTH && y < GPU_IMAGE_HEIGHT) {
        int pixelIndex = (y * GPU_IMAGE_WIDTH + x) * 3;
        Ray3D mainRay = getInitialRay(x,y);
        GPU_RAW_IMAGE_DATA[pixelIndex] = mainRay.directionX;
        GPU_RAW_IMAGE_DATA[pixelIndex+1] = mainRay.directionY;
        GPU_RAW_IMAGE_DATA[pixelIndex+2] = mainRay.directionZ;
    }
}

__global__ void convertRawImage(float* GPU_RAW_IMAGE_DATA, unsigned char* GPU_IMAGE_DATA) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < GPU_IMAGE_WIDTH && y < GPU_IMAGE_HEIGHT) {
        int pixelIndex = (y * GPU_IMAGE_WIDTH + x) * 3;
        GPU_IMAGE_DATA[pixelIndex] = static_cast<unsigned char>(255.99f * fminf(1.0,GPU_RAW_IMAGE_DATA[pixelIndex]));
        GPU_IMAGE_DATA[pixelIndex+1] = static_cast<unsigned char>(255.99f * fminf(1.0,GPU_RAW_IMAGE_DATA[pixelIndex+1]));
        GPU_IMAGE_DATA[pixelIndex+2] = static_cast<unsigned char>(255.99f * fminf(1.0,GPU_RAW_IMAGE_DATA[pixelIndex+2]));
    }
}





void Shader::initShader(const int& imageWidth, const int& imageHeight) {
    Shader::threadsPerBlock = {16,8};
    Shader::blocksPerGrid = {static_cast<unsigned int>(ceil(imageWidth / static_cast<float>(threadsPerBlock.x))),static_cast<unsigned int>(ceil(imageHeight / static_cast<float>(threadsPerBlock.y)))};
    //Set GPU Data:
    cudaMemcpyToSymbol(GPU_IMAGE_WIDTH,&imageWidth,sizeof(int));
    cudaMemcpyToSymbol(GPU_IMAGE_HEIGHT,&imageHeight,sizeof(int));
    Shader::allocateMemory();
}


void Shader::allocateMemory() {
    int imageWidth;
    int imageHeight;
    cudaMemcpyFromSymbol(&imageWidth,GPU_IMAGE_WIDTH,sizeof(int));
    cudaMemcpyFromSymbol(&imageHeight,GPU_IMAGE_HEIGHT,sizeof(int));
    //Frees any Dynamic Memory
    Shader::freeDynamicMemory();
    //Re-Allocates Dynamic Memory Based on New Size
    //GPU DYNAMIC MEMORY ALLOCATION
    cudaMalloc(&Shader::GPU_RAW_IMAGE_DATA,imageWidth * imageHeight * 3 * sizeof(float));
    cudaMalloc(&Shader::GPU_IMAGE_DATA,imageWidth * imageHeight * 3 * sizeof(unsigned char));
    //CPU DYNAMIC MEMORY ALLOCATION
    Shader::hostImageData = new unsigned char[imageWidth * imageHeight * 3];
}


void Shader::freeDynamicMemory() {
    //Frees GPU DYNAMIC MEMORY
    if(Shader::GPU_RAW_IMAGE_DATA != nullptr) {
        cudaFree(Shader::GPU_RAW_IMAGE_DATA);
    }

    if(Shader::GPU_IMAGE_DATA != nullptr) {
        cudaFree(Shader::GPU_IMAGE_DATA);
    }

    //FREES CPU DYNAMIC MEMORY
    if(Shader::hostImageData != nullptr) {
        delete[] Shader::hostImageData;
        Shader::hostImageData = nullptr;
    }
}

void Shader::updateCameraData(float* cameraData) {
    cudaMemcpyToSymbolAsync(GPU_CAMERA_POS_X,&cameraData[0],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[0]);
    cudaMemcpyToSymbolAsync(GPU_CAMERA_POS_Y,&cameraData[1],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[1]);
    cudaMemcpyToSymbolAsync(GPU_CAMERA_POS_Z,&cameraData[2],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[2]);
    cudaMemcpyToSymbolAsync(GPU_DEPTH_OF_FIELD_STRENGTH,&cameraData[3],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[3]);
    cudaMemcpyToSymbolAsync(GPU_FOCAL_DISTANCE,&cameraData[4],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[4]);
    cudaMemcpyToSymbolAsync(GPU_FOV_SCALAR,&cameraData[5],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[5]);
    cudaMemcpyToSymbolAsync(GPU_COSX,&cameraData[6],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[6]);
    cudaMemcpyToSymbolAsync(GPU_COSY,&cameraData[7],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[7]);
    cudaMemcpyToSymbolAsync(GPU_SINX,&cameraData[8],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[8]);
    cudaMemcpyToSymbolAsync(GPU_SINY,&cameraData[9],sizeof(float),0,cudaMemcpyHostToDevice,GPU_STREAMS[9]);
    cudaDeviceSynchronize();
}

void Shader::updateWorldData(float*& newWorldData, size_t& sizeOfWorldData) {
    cudaMemcpyToSymbol(GPU_WORLD_DATA, newWorldData, sizeof(float) * sizeOfWorldData);
    cudaMemcpyToSymbol(GPU_WORLD_SIZE,&sizeOfWorldData,sizeof(size_t));
}

void Shader::renderFrame() {
    mainShader<<<Shader::blocksPerGrid,Shader::threadsPerBlock>>>(Shader::GPU_RAW_IMAGE_DATA,Shader::GPU_IMAGE_DATA);
    convertRawImage<<<Shader::blocksPerGrid,Shader::threadsPerBlock>>>(Shader::GPU_RAW_IMAGE_DATA,Shader::GPU_IMAGE_DATA);
}

void Shader::retrieveImageData() {
    int imageWidth;
    int imageHeight;
    cudaMemcpyFromSymbolAsync(&imageWidth,GPU_IMAGE_WIDTH,sizeof(int));
    cudaMemcpyFromSymbolAsync(&imageHeight,GPU_IMAGE_HEIGHT,sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(Shader::hostImageData,Shader::GPU_IMAGE_DATA, imageWidth * imageHeight * 3 * sizeof(unsigned char),cudaMemcpyDeviceToHost);
}

void Shader::saveImage(const std::string& fileName) {
    int imageWidth;
    int imageHeight;
    cudaMemcpyFromSymbolAsync(&imageWidth,GPU_IMAGE_WIDTH,sizeof(int));
    cudaMemcpyFromSymbolAsync(&imageHeight,GPU_IMAGE_HEIGHT,sizeof(int));
    cudaDeviceSynchronize();
    unsigned char* tempImageData = new unsigned char[imageWidth * imageHeight * 3];
    Shader::retrieveImageData();
    // Write image data to BMP file
    int pixelIndex;
    //Swaps RGB Values For SDL Expectations
    for (int y = 0; y < imageHeight; ++y) {
        for (int x = 0; x < imageWidth; ++x) {
            pixelIndex = (y*imageWidth + x)*3;
            tempImageData[pixelIndex] = Shader::hostImageData[pixelIndex+2];
            tempImageData[pixelIndex+1] = Shader::hostImageData[pixelIndex+1];
            tempImageData[pixelIndex+2] = Shader::hostImageData[pixelIndex];
        }
    }
    std::cout<<"saving image..."<<std::endl;
    // Write image data to BMP file
    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(tempImageData, imageWidth, imageHeight, 24, imageWidth * 3, 0xFF0000, 0x00FF00, 0x0000FF, 0);
    std::string filePath = "./Saved Images/" +fileName + ".bmp";
    std::cout<<"FilePath: "<<filePath<<std::endl;
    SDL_SaveBMP(surface,filePath.c_str());
}