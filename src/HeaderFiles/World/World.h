#ifndef WORLD_H
#define WORLD_H

#include "./Camera.h"
#include "./Sphere.h"
#include "./Triangle.h"
#include <vector>
#include <string>


class World {

public:
    World(std::string worldName);
    void addMaterial( Material* newMaterial);
    void addMaterial( Vec3 primaryColor,  Vec3 lightColor,  float luminance,  float smoothness,  float glossiness,  float refractiveIndex);
    void addTriangle( Triangle* newTriangle);
    void addTriangle( Vec3 point1,  Vec3 point2,  Vec3 point3,  int worldMaterialIndex);

    float* getWorldTriangles();
    size_t getWorldTriangleDataSize();
    float* getWorldMaterials();

private:
    std::string worldName;
    std::vector<Material*> worldMaterials;
    int numMaterials;
    std::vector<Triangle*> worldTriangles;
    int numTriangles;


};

#endif