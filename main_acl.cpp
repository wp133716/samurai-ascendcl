#include <iostream>
#include <opencv2/opencv.hpp>
#include "sam2_tracker_acl.h"

std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 0, 255),     // red 0
    cv::Scalar(0, 255, 0),     // green 1
    cv::Scalar(255, 0, 0),     // blue 2
    cv::Scalar(255, 255, 0),   // cyan 3
    cv::Scalar(255, 0, 255),   // magenta 4
    cv::Scalar(0, 255, 255),   // yellow 5
    cv::Scalar(255, 255, 255), // white 6
    cv::Scalar(128, 128, 128), // gray 7
    cv::Scalar(40, 50, 174),
    cv::Scalar(128, 0, 0),
    cv::Scalar(128, 128, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(128, 0, 128),
    cv::Scalar(0, 128, 128),
    cv::Scalar(0, 0, 128),
    cv::Scalar(0, 0, 0)
};

int main(int argc, char** argv) {
    std::string modelPath = "../om_model";
    SAM2TrackerAcl tracker(modelPath);

    std::string videoPath = "../data/1917.mp4";
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()){
        std::cerr << "Error: cannot open video file : " << videoPath << std::endl;
        return -1;
    }

    // 获取videoPath的文件名
    std::string videoName = videoPath.substr(videoPath.find_last_of("/\\") + 1);  // 1917-1.mp4
    // videoName = videoName.substr(0, videoName.find_last_of(".")); // 1917-1
    // std::cout << "videoName: " << videoName << std::endl;
    std::cout << "start tracking video: " << videoName << std::endl;
    // cv::namedWindow(videoName, cv::WINDOW_NORMAL);

    int numframes = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // cv VideoWriter_fourcc
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(frameWidth, frameHeight));

    int frameIdx = 0;
    cv::Mat frame;
    cv::Mat predMask;
    auto start = std::chrono::high_resolution_clock::now();
    while(cap.read(frame)){
        auto startStep = std::chrono::high_resolution_clock::now();
        std::cout << "\033[32mframeIdx: " << frameIdx << "\033[0m" << std::endl;
        if(frameIdx == 0){
            // cv select roi
            cv::Rect firstBbox = cv::selectROI(videoName, frame);
            // cv::Rect firstBbox(384, 304, 342, 316);
            std::cout << "first_bbox (x, y, w, h): " << firstBbox.x << ", " << firstBbox.y << ", " << firstBbox.width << ", " << firstBbox.height << std::endl;
            predMask = tracker.addFirstFrameBbox(frameIdx, frame, firstBbox);
        } else {
            predMask = tracker.trackStep(frameIdx, frame);
        }
        cv::resize(predMask, predMask, cv::Size(frameWidth, frameHeight));

        // 结果可视化与保存
        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8UC1, 255);

        cv::Mat maskImg = cv::Mat::zeros(frame.size(), CV_8UC3);
        maskImg.setTo(colors[8], binaryMask);
        cv::addWeighted(frame, 1, maskImg, 0.9, 0, frame);

        // std::vector<int> bbox = {0, 0, 0, 0};
        cv::Rect bbox(0, 0, 0, 0);
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(binaryMask, nonZeroPoints);
        if (!nonZeroPoints.empty()) {
            bbox = cv::boundingRect(nonZeroPoints);
        }
        // std::cout << "bbox (x, y, w, h): " << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << std::endl;
        cv::rectangle(frame, bbox, colors[8], 1);

        cv::imshow(videoName, frame);
        // cv::imwrite(std::to_string(frameIdx) + ".jpg", frame);
        writer.write(frame);
        cv::waitKey(1);
        frameIdx++;
        auto durationStep = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startStep);
        std::cout << "step spent: " << durationStep.count() << " ms" << std::endl;
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "total spent: " << duration.count() << " ms" << std::endl;
    std::cout << "every frame spent: " << duration.count() / frameIdx << " ms" << std::endl;
    std::cout << "FPS: " << frameIdx / (duration.count() / 1000.0) << std::endl;

    cap.release();

    return 0;
}