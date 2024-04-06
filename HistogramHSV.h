
#include <vector>
#pragma once
#define HISTOGRAM_HSV
#ifdef HISTOGRAM_HSV

struct Bin
{
    long double pixel_amount = 0;
};

class Histogram
{
public:
    int min_intensity;
    int max_intensity;
    long total_pixels;
    std::vector<Bin> histogram_intensity_bar;
    Histogram(int min_intensity, int max_intensity)
    {
        for (int i = 0; i <= max_intensity; i++)
        {
            histogram_intensity_bar.push_back(Bin());
        }
        this->max_intensity = max_intensity;
        this->min_intensity = min_intensity;
        total_pixels = 1;
    }

    void normalize()
    {
        long total_pixels = getTotalPixels();
        if (total_pixels > 0)
        {
            for (auto& bin : histogram_intensity_bar)
            {
                bin.pixel_amount /= total_pixels;
            }
        }
    }

    long getTotalPixels()
    {
        return total_pixels;
    }
};

class HistogramHSV
{
public:
    Histogram* histogram_hchannel;
    Histogram* histogram_schannel;
    Histogram* histogram_vchannel;
    int min_intensity;
    int max_intensity;

    HistogramHSV(int min_intensity, int max_intensity)
    {
        this->max_intensity = max_intensity;
        this->min_intensity = min_intensity;
        histogram_hchannel = new Histogram(min_intensity, max_intensity);
        histogram_schannel = new Histogram(min_intensity, max_intensity);
        histogram_vchannel = new Histogram(min_intensity, max_intensity);
    }
};

#endif // DEBUG

