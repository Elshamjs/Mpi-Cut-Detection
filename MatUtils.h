#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#pragma once
#define MAT_UTILS
#ifdef MAT_UTILS

static class MatUtils
{
public:
	static cv::Mat toColorHSV(cv::Mat &input)
	{
		cv::Mat output;
		cv::cvtColor(input, output, cv::COLOR_BGR2HSV);
		return output;
	}
	static cv::Mat normalizeHSV(cv::Mat &input)
	{
		cv::Mat output;
		cv::normalize(input, output, 0, 255, cv::NORM_MINMAX);
		return output;
	}
	static cv::Mat calculateHistogram(cv::Mat &image)
	{
		cv::Mat output;
		static int channels[] = { 0, 1, 2 };
		static int histSize[] = { 256, 256, 256 };
		static float hranges[] = { 0, 255 }; 
		static float sranges[] = { 0, 255 }; 
		static float vranges[] = { 0, 255 };
		static const float* ranges[] = { hranges, sranges, vranges };
		cv::calcHist(&image, 1, channels, cv::Mat(), output, 3, histSize, ranges);
		return output;
	}
	static double calculateBhattacharyyaDistance(cv::Mat &hist1, cv::Mat hist2)
	{
		cv::Mat out1, out2;
		cv::normalize(hist1, out1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		cv::normalize(hist2, out2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		double distance = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
		return distance;
	}
};

#endif // MAT_HISTOGRAM_HSV

