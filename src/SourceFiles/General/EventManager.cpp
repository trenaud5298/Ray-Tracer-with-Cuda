#include "../../HeaderFiles/General/EventManager.h"

#define MAIN_MENU 1
#define RUNNING 2
#define PAUSED 3
#define SETTINGS_MENU 4
#define EDIT_MENU 5
#define WORLD_EDITTOR 6
#define QUIT_PROGRAM 7


EventManager::EventManager(FileParser* mainFileParser,Settings* mainSettings) :
mainFileParser(mainFileParser),mainSettings(mainSettings) {
    this->initializeProgram();
}

EventManager::~EventManager() {

}

void EventManager::initializeProgram() {
    World* testWorld = new World("test");
    testWorld->addTriangle(Vec3(1.0,1.0,0.0),Vec3(0.0,1.0,0.0),Vec3(1.0,0.0,0.0),0);
    // testWorld->addTriangle(Vec3(1.0,1.0,1.0),Vec3(1.0,1.0,1.0),Vec3(1.0,1.0,1.0),1.0);
    // testWorld->addTriangle(Vec3(1.0,1.0,1.0),Vec3(1.0,1.0,1.0),Vec3(1.0,1.0,1.0),2.0);

    float* testArray = testWorld->getWorldTriangles();
    size_t worldDataSize = testWorld->getWorldTriangleDataSize();


    Shader::initShader(1920,1080);
    Camera* mainCam = new Camera();
    mainCam->rotateCamera(30,40);
    Shader::updateCameraData(mainCam->toArray());
    Shader::updateWorldData(testArray,worldDataSize);
    Shader::renderFrame();
    Shader::saveImage("test");
}



void EventManager::runProgram() {
    while(this->programState != QUIT_PROGRAM) {
        this->handleEvents();
        this->renderScene();
    }
    delete this;
}

void EventManager::handleEvents() {

}

void EventManager::renderScene() {

}