#include "../../HeaderFiles/World/Material.h"

Material::Material(Vec3 primaryColor, Vec3 lightColor, float luminance, float smoothness, float glossiness, float refractiveIndex) :
    primaryColor(primaryColor), lightColor(lightColor), luminance(luminance), smoothness(smoothness), glossiness(glossiness), refractiveIndex(refractiveIndex) {

}

