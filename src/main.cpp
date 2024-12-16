#include <iostream>
#include <opencv2/opencv.hpp>

#include "inference.h"


int main()
{
    std::string img_path = "./images/vlcsnap.png";

    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return 1;
    }
    infer::run_inference(img);
    cv::imshow("img", img);
    cv::waitKey();
    return 0;
}
