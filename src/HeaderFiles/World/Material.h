#ifndef MATERIAL_H
#define MATERIAL_H

#include "../Math/Vec3.h"
#include <string>



struct Material {
    Material(Vec3 primaryColor, Vec3 lightColor, float luminance, float smoothness, float glossiness, float refractiveIndex);
    //Total Size:70 Bytes
    uint8_t materialType;
    Vec3 primaryColor;
    Vec3 secondaryColor;
    Vec3 lightColor;
    bool useDifferentSpecularHightlightColor;
    Vec3 specularHighLightColor;
    float luminance;
    float smoothness;
    float glossiness;
    float refractiveIndex;
    //Optional | Subject TO Change:
    uint8_t textureID;
    uint8_t blankVal1;
    uint8_t blankVal2;
    uint8_t blankVal3;
};

#endif