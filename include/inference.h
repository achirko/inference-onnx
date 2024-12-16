#pragma once
#include "constants.h"
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include "utils/augment.h"
#include "utils/common.h"

namespace infer
{
  std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels);

  void plot_results(cv::Mat& img, std::vector<YoloResults>& results,
                    std::vector<cv::Scalar> color, std::unordered_map<int, std::string>& names);

  void run_inference(cv::Mat& img, const std::string& modelPath = "./models/YOLO.onnx",
                    float conf_threshold = 0.1f, float iou_threshold = 0.4f);
};
