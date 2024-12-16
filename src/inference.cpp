#include <random>

#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"

#include "inference.h"


cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    cv::Scalar color;
    for (int i = 0; i < numChannels; i++) {
        color[i] = dis(gen); // for each channel separately generate value
    }

    return color;
}

std::vector<cv::Scalar> infer::generateRandomColors(int class_names_num, int numChannels) {
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < class_names_num; i++) {
        cv::Scalar color = generateRandomColor(numChannels);
        colors.push_back(color);
    }
    return colors;
}

void infer::plot_results(cv::Mat& img, std::vector<YoloResults>& results,
                         std::vector<cv::Scalar> color, std::unordered_map<int, std::string>& names)
{
    int radius = 5;
    bool drawLines = true;

    auto raw_image_shape = img.size();
    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;
        int color_num = res.class_idx;

        // Draw bounding box
        rectangle(img, res.bbox, color[res.class_idx], 2);

        // Try to get the class name corresponding to the given class_idx
        std::string class_name;
        auto it = names.find(res.class_idx);
        if (it != names.end()) {
            class_name = it->second;
        }
        else {
            std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
            // Then convert it to a string anyway
            class_name = std::to_string(res.class_idx);
        }

        // Create label
        std::stringstream labelStream;
        labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
        std::string label = labelStream.str();

        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
        cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rect_to_fill, color[res.class_idx], -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    }
}


void infer::run_inference(cv::Mat& img, const std::string& modelPath, float conf_threshold, float iou_threshold)
{
    const std::string& onnx_provider = OnnxProviders::CPU; // "cpu";
    const std::string& onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;
    int conversion_code = cv::COLOR_BGR2RGB;
    int num_threads = 4;

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str(), num_threads);
    std::vector<YoloResults> objs = model.predict_once(img, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    std::vector<cv::Scalar> colors = infer::generateRandomColors(model.getNc(), model.getCh());
    std::unordered_map<int, std::string> names = model.getNames();

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    infer::plot_results(img, objs, colors, names);
}
