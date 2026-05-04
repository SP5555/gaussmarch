#pragma once
#include <vector>
#include "../layers/layer.h"

class Pipeline
{
public:
    void add(Layer *layer) { layers.push_back(layer); }
    void clear() { layers.clear(); }

    void zeroGrad()
    {
        for (auto *l : layers)
            l->zeroGrad();
    }

    void forward()
    {
        for (auto *l : layers)
            l->forward();
    }

    void backward()
    {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            (*it)->backward();
    }

private:
    std::vector<Layer*> layers;
};