#include "../../HeaderFiles/World/Triangle.h"

Triangle::Triangle(Vec3 pointA, Vec3 pointB, Vec3 pointC, int worldMaterialIndex) : 
    pointA(pointA),pointB(pointB),pointC(pointC),worldMaterialIndex(worldMaterialIndex) {

}

Triangle::Triangle(float pointAX, float pointAY, float pointAZ, float pointBX, float pointBY, float pointBZ, float pointCX, float pointCY, float pointCZ, int worldMaterialIndex) :
    pointA(Vec3(pointAX,pointAY,pointAZ)),pointB(Vec3(pointBX,pointBY,pointBZ)),pointC(Vec3(pointCX,pointCY,pointCZ)) {
}

