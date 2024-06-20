#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "../Math/Vec3.h"
#include "./Material.h"

class World;

class Triangle {

public:
    Triangle(Vec3 pointA, Vec3 pointB, Vec3 pointC, int worldMaterialIndex);
    Triangle(float pointAX, float pointAY, float pointAZ, float pointBX, float pointBY, float pointBZ, float pointCX, float pointCY, float pointCZ, int worldMaterialIndex);

    Vec3 getNormalVector();
private:
    friend class World;
    Vec3 pointA;
    Vec3 pointB;
    Vec3 pointC;
    int worldMaterialIndex;


};


#endif