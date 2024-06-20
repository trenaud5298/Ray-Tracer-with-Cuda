#ifndef DISPLAY_H
#define DISPLAY_H

#include "../../../lib/SDL/include/SDL.h"

class Display {

public:
    Display(int resolutionX, int resolutionY);
    int resolutionX;
    int resolutionY;
    SDL_Window* displayWindow;
    SDL_Renderer* imageRenderer;
    SDL_Texture* imageTexture;

protected:


private:


};

#endif