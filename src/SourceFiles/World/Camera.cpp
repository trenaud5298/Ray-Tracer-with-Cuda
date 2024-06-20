#include "../../HeaderFiles/World/Camera.h"

const static double M_PI = 3.1415926535897932384626433832795028841;

const static int FOCAL_TYPE_DISTANCE = 0;
const static int FOCAL_TYPE_POINT = 1;


Camera::Camera() : position(Vec3()), focalType(FOCAL_TYPE_DISTANCE), focalDistance(1.0f), fieldOfView(90),depthOfFieldStrength(0), horizontalRotation(0), verticalRotation(0) {
    this->updateRotationData();
}

float* Camera::toArray() {
    float* cameraDataArray = new float[10];
    cameraDataArray[0] = this->position.x;
    cameraDataArray[1] = this->position.y;
    cameraDataArray[2] = this->position.z;
    cameraDataArray[3] = this->depthOfFieldStrength;
    cameraDataArray[4] = this->focalDistance;
    cameraDataArray[5] = static_cast<float>(focalDistance*(std::tan(M_PI*this->fieldOfView/360)));
    cameraDataArray[6] = this->cosX;
    cameraDataArray[7] = this->cosY;
    cameraDataArray[8] = this->sinX;
    cameraDataArray[9] = this->sinY;
    std::cout<<"Data: "<<std::endl;
    for(int i = 0; i < 10; i++) {
        std::cout<<"    Index: "<<i<<" Value: "<<cameraDataArray[i]<<std::endl;
    }
    return cameraDataArray;
}

void Camera::rotateCamera(float horizontalRotation, float verticalRotation) {
    this->horizontalRotation += horizontalRotation;
    float newRotation = this->verticalRotation + verticalRotation;
    this->verticalRotation = (newRotation < -89) ? -89 : (newRotation > 89) ? 89 : newRotation;
    this->updateRotationData();
}

void Camera::updateRotationData() {
    this->cosX = std::cos(M_PI*this->horizontalRotation/180);
    this->cosY = std::cos(M_PI*-this->verticalRotation/180);
    this->sinX = std::sin(M_PI*this->horizontalRotation/180);
    this->sinY = std::sin(M_PI*-this->verticalRotation/180);
}

void Camera::moveForward(float distance) {
    /*X-coord*/this->position.x += distance*this->sinX;
    /*Z-coord*/this->position.z += distance*this->cosX;

}
void Camera::moveBackward(float distance) {
    /*X-coord*/this->position.x -= distance*this->sinX;
    /*Z-coord*/this->position.z -= distance*this->cosX;

}
void Camera::moveRight(float distance) {
    /*X-coord*/this->position.x += distance*this->cosX;
    /*Z-coord*/this->position.z -= distance*this->sinX;

}
void Camera::moveLeft(float distance) {
    /*X-coord*/this->position.x -= distance*this->cosX;
    /*Z-coord*/this->position.z += distance*this->sinX;

}
void Camera::moveUp(float distance) {
    /*Y-coord*/this->position.y += distance;

}
void Camera::moveDown(float distance) {
    /*Y-coord*/this->position.y -= distance;

}
