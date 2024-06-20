#include "../../HeaderFiles/World/World.h"


World::World(std::string worldName) : worldName(worldName), numTriangles(0), numMaterials(0) {
    //Creates A Default White Material:
    this->worldMaterials.push_back(new Material(Vec3(1.0,1.0,1.0),Vec3(0.0,0.0,0.0),0.0,0.0,0.0,0.0));
    this->numMaterials+=1;
}


void World::addMaterial(Material* newMaterial) {
    this->worldMaterials.push_back(newMaterial);
    this->numMaterials+=1;
}

void World::addMaterial(Vec3 primaryColor,  Vec3 lightColor,  float luminance,  float smoothness,  float glossiness,  float refractiveIndex) {
    this->worldMaterials.push_back(new Material(primaryColor,lightColor,luminance,smoothness,glossiness,refractiveIndex));
    this->numMaterials+=1;
}

void World::addTriangle(Triangle* newTriangle) {
    this->worldTriangles.push_back(newTriangle);
    this->numTriangles+=1;
}

void World::addTriangle(Vec3 point1,  Vec3 point2,  Vec3 point3,  int worldMaterialIndex) {
    this->worldTriangles.push_back(new Triangle(point1,point2,point3,worldMaterialIndex));
    this->numTriangles+=1;
}

float* World::getWorldTriangles() {
    float* triangleArray = new float[23*this->numTriangles+1];
    //First Index is Number of Triangles
    triangleArray[0] = this->numTriangles;
    for(int i = 1; i < 23*this->numTriangles+1; i+=23) {
        std::cout<<"Index Start Val: "<<i<<std::endl;
        Triangle* currentTriangle = this->worldTriangles[static_cast<int>((i-1)/23)];
        triangleArray[i] = currentTriangle->worldMaterialIndex;
        std::cout<<"Triangle Index: "<<(i/23)<<"   Val 1: "<<triangleArray[i]<<std::endl;
        Vec3 AB = currentTriangle->pointB - currentTriangle->pointA;
        Vec3 AC = currentTriangle->pointC - currentTriangle->pointA;
        Vec3 BC = currentTriangle->pointC - currentTriangle->pointB;
        Vec3 normVector = AB.cross(AC);
        normVector.normalize();
        float distanceToOrigin = normVector.dot(currentTriangle->pointA);
        std::cout<<"EQ OF PLANE: "<<normVector.x<<"x + "<<normVector.y<<"y + "<<normVector.z<<"z = "<<distanceToOrigin<<std::endl;
        triangleArray[i+1] = normVector.x;
        triangleArray[i+2] = normVector.y;
        triangleArray[i+3] = normVector.z;
        triangleArray[i+4] = distanceToOrigin;
        Vec3 MidpointAB = currentTriangle->pointA+currentTriangle->pointB;
        MidpointAB*0.5f;

        Vec3 MidpointAC = currentTriangle->pointA+currentTriangle->pointC;
        MidpointAB*0.5f;

        Vec3 MidpointBC = currentTriangle->pointB+currentTriangle->pointC;
        MidpointAB*0.5f;
        triangleArray[i+5] = 1.0;
        triangleArray[i+6] = 1.0;
        triangleArray[i+7] = 1.0;
        triangleArray[i+8] = 1.0;
        triangleArray[i+9] = 1.0;
        triangleArray[i+10] = 1.0;
        triangleArray[i+11] = 1.0;
        triangleArray[i+12] = 1.0;
        triangleArray[i+13] = 1.0;
        triangleArray[i+14] = 1.0;
        triangleArray[i+15] = 1.0;
        triangleArray[i+16] = 1.0;
        triangleArray[i+17] = 1.0;
        triangleArray[i+18] = 1.0;
        triangleArray[i+19] = 1.0;
        triangleArray[i+20] = 1.0;
        triangleArray[i+21] = 1.0;
        triangleArray[i+22] = 1.0;
    }
    std::cout<<"Testing"<<std::endl;
    for(int i =0; i<23*this->numTriangles; i++) {
        if(triangleArray[i]!=1.0) {
            std::cout<<"Error at Index: "<<i<<std::endl;
        }
    }
    return triangleArray;
}

size_t World::getWorldTriangleDataSize() {
    return static_cast<size_t>(this->numTriangles*23+1);
}