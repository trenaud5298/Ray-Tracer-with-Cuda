#ifndef EVENTMANAGER_H
#define EVENTMANAGER_H

#include "../World/World.h"
#include "./Settings.h"
#include "./FileParser.h"
#include "../../HeaderFiles/RayTracing/Shader.cuh"

class EventManager {

public:
    EventManager(FileParser* mainFileParser, Settings* mainSettings);
    void runProgram();
protected:

private:
    FileParser* mainFileParser;
    Settings* mainSettings;
    short programState;
    void initializeProgram();
    void handleEvents();
    void renderScene();
    ~EventManager();
};

#endif