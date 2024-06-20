#include <iostream>
#include "./HeaderFiles/General/Settings.h"
#include "./HeaderFiles/General/FileParser.h"
#include "./HeaderFiles/General/EventManager.h"
#include <filesystem>
#include "./HeaderFiles/Math/Vec3.h"
#include <thread>
#include <chrono>

#undef main

int main(int argc, char* argv[]) {
    std::cout<<"Hello World"<<std::endl;
    //Creates File Parse to Get all Important Files
    FileParser* mainFileReader = new FileParser(std::filesystem::current_path(), false);
    //Creates Settings Menu To Read/Write Ini File
    Settings* mainSettings = new Settings(mainFileReader->configFile.string(), mainFileReader->getDirectoryPathString("Settings"));
    
    EventManager* mainProgram = new EventManager(mainFileReader,mainSettings);
    

    return 0; 
}
