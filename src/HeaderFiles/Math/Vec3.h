#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <sstream>
#include <string>
#include <iostream>

struct Vec3 {
    float x,y,z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {};

    //General Operations:
    float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    void normalize() {
        float inverseLength = 1/sqrtf(x*x + y*y + z*z);
        x *= inverseLength;
        y *= inverseLength;
        z *= inverseLength;
    }

    std::string string() {
        std::stringstream vecAsString;
        vecAsString << "( " << x << ", " << y << ", " << z << " )";
        return vecAsString.str();
    }
    

    //Operations Between Float and Vec3:
    Vec3 operator+(const float& scalar) const {
        return Vec3(x+scalar,y+scalar,z+scalar);
    }
    Vec3 operator-(const float& scalar) const {
        return Vec3(x-scalar,y-scalar,z-scalar);
    }
    Vec3 operator*(const float& scalar) const {
        return Vec3(x*scalar,y*scalar,z*scalar);
    }
    Vec3 operator/(const float& scalar) const {
        return Vec3(x/scalar,y/scalar,z/scalar);
    }


    //Operations Between Two Vec3:
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x+other.x, y+other.y, z+other.z);
    }
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x-other.x, y-other.y, z-other.z);
    }
    Vec3 operator*(const Vec3& other) const {
        return Vec3(x*other.x, y*other.y, z*other.z);
    }
    Vec3 operator/(const Vec3& other) const {
        return Vec3(x/other.x, y/other.y, z/other.z);
    }

    //Special Operations Between Vec3's:
    float dot(const Vec3& other) const {
        return (x*other.x + y*other.y + z*other.z);
    }

    Vec3 cross(const Vec3& other) const {
        return Vec3(y*other.z-z*other.y,z*other.x-x*other.z,x*other.y-y*other.x);
    }

    float crossLength(const Vec3& other) const {
        return Vec3(y*other.z-z*other.y,z*other.x-x*other.z,x*other.y-y*other.x).length();
    }

    float angleBetween(const Vec3& other) const {
        return acosf( this->dot(other) / (this->length() * other.length()));
    }

};

inline std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}



#endif