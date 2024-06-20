#ifndef CAMERA_H
#define CAMERA_H

#include "../Math/Vec3.h"

class Camera {


private:
    Vec3 position;
    float depthOfFieldStrength;
    int focalType;
    Vec3 focalPoint;
    float focalDistance;
    int fieldOfView;
    float horizontalRotation;
    float verticalRotation;
    float cosX;
    float cosY;
    float sinX;
    float sinY;

public:
    Camera();
    void setFocalPoint(float x, float y, float z);
    void setFocalPoint(Vec3 focalPoint);
    float* toArray();

    //Movement Functions
    void moveForward(float distance);
    void moveBackward(float distance);
    void moveRight(float distance);
    void moveLeft(float distance);
    void moveUp(float distance);
    void moveDown(float distance);

    //Depth OF Field Methods
    void increaseDepthOfFieldStrength(float amount);
    void decreaseDepthOfFieldStrength(float amount);
    void incrimentFocalDistance(float amount);
    void decreaseFocalDistance(float amount);

    void rotateCamera(float horizontalRotation, float verticalRotation);


private:
    void updateRotationData();
};


#endif