#pragma once
#include <opencv2/core.hpp>
using namespace cv;

class Ridge
{
public:
	void addPoint(Point newPoint) {
		points.push_back(newPoint);
	};
	Mat getPoints() {
		return points;
	}
private:
		Mat points;
};
