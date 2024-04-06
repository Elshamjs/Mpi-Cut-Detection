#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "HistogramHSV.h"
#pragma once
#define MAT_FRAME_HISTOGRAM
#ifdef MAT_FRAME_HISTOGRAM

class MatFrameHistogram : public cv::Mat
{
public:
    HistogramHSV* histogram_hsv = nullptr;

    MatFrameHistogram() : cv::Mat()
    {

    }

    void reloadHistogram()
    {
        histogram_hsv = new HistogramHSV(0, 255);
        cv::Vec3b pixel;
        for (int m = 0; m < this->rows; m++)
        {
            for (int n = 0; n < this->cols; n++)
            {
                pixel = this->at<cv::Vec3b>(m, n);
                histogram_hsv->histogram_hchannel->histogram_intensity_bar[(uchar)pixel[0]].pixel_amount++;
                histogram_hsv->histogram_schannel->histogram_intensity_bar[(uchar)pixel[1]].pixel_amount++;
                histogram_hsv->histogram_vchannel->histogram_intensity_bar[(uchar)pixel[2]].pixel_amount++;
            }
        }
        histogram_hsv->histogram_hchannel->total_pixels = this->rows * this->cols;
        histogram_hsv->histogram_schannel->total_pixels = this->rows * this->cols;
        histogram_hsv->histogram_vchannel->total_pixels = this->rows * this->cols;

        histogram_hsv->histogram_hchannel->normalize();
        histogram_hsv->histogram_schannel->normalize();
        histogram_hsv->histogram_vchannel->normalize();
        
    }

    HistogramHSV* getHistogramHSV()
    {
        if (histogram_hsv == nullptr)
        {
            reloadHistogram();
            return histogram_hsv;
        }
        else
        {
            return histogram_hsv;
        }
    }
};


#endif 
