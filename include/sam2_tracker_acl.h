#ifndef SAM2_TRACKER_ACL_H
#define SAM2_TRACKER_ACL_H

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <thread>
#include <chrono>
#include <omp.h>
#include <unistd.h>
#include <fstream>
#include <acl/acl.h>

#include "kalman_filter.h"

#define ACL_CHECK(func)                                                                                                                   \
    do                                                                                                                                    \
    {                                                                                                                                     \
        aclError ret = (func);                                                                                                            \
        if (ret != ACL_SUCCESS)                                                                                                           \
        {                                                                                                                                 \
            std::cerr << "aclError in \"" << #func << "\" at " << __FILE__ << ":" << __LINE__ << " with error code " << ret << std::endl; \
        }                                                                                                                                 \
    } while (0)

struct MemoryBankEntry
{
    std::vector<float> maskmem_features;
    std::vector<float> maskmem_pos_enc;
    std::vector<float> obj_ptr;
    float best_iou_score;
    float obj_score_logits;
    float kf_score;
};

struct PostprocessResult
{
    int bestIoUIndex;
    float bestIouScore;
    float kfScore;
};

class SAM2TrackerAcl
{
public:
    SAM2TrackerAcl() {}
    SAM2TrackerAcl(const std::string &modelPath);
    ~SAM2TrackerAcl();

    void loadNetwork(const std::string &modelPath);

    cv::Mat addFirstFrameBbox(int frameIdx, const cv::Mat &firstFrame, const cv::Rect &bbox);

    cv::Mat trackStep(int frameIdx, const cv::Mat &frame);

    void imageEncoderInference(const std::vector<float> &frame, std::vector<std::vector<float>> &imageEncoderOutput);
    void memoryAttentionInference(int frameIdx,
                                  const std::vector<float> &visionFeatures,
                                  std::vector<std::vector<float>> &memoryAttentionOutputs);
    void maskDecoderInference(const std::vector<float> &inputPoints,
                              const std::vector<int>   &inputLabels,
                              const std::vector<float> &pixFeatWithMem,
                              const std::vector<float> &highResFeatures0,
                              const std::vector<float> &highResFeatures1,
                              std::vector<std::vector<float>> &maskDecoderOutputs);
    void memoryEncoderInference(const std::vector<float> &visionFeatures,
                                const std::vector<float> &highResMasksForMem,
                                const std::vector<float> &objectScoreLogits,
                                bool isMaskFromPts,
                                std::vector<std::vector<float>> &memoryEncoderOutputs);

    void preprocessImage(const cv::Mat &src, std::vector<float> &dest);
    PostprocessResult postprocessOutput(const std::vector<std::vector<float>> &maskDecoderOutputs);

    void test_memoryAttentionInference(std::vector<std::vector<float>> &memoryAttentionOutputs);

private:
    void initAscendCL();

    void printDataType(aclDataType type);

    void addDataset(const void *hostData, size_t bufferSize, aclmdlDataset *dataset);

    void createModelOutput(int modelDescIdx, aclmdlDataset *outputDataset);

    void retrieveOutputData(aclmdlDataset *outputDataset, size_t numOutputNodes, std::vector<std::vector<float>> &outputData);

    void releaseResources(aclmdlDataset *dataset, int dataSize);
    void unloadModel();
    void finalizeACL();

    std::vector<float> readBinaryFile(const std::string &filePath);

    // Ascend related
    int32_t _deviceId = 0;
    std::vector<uint32_t> _modelIds{4};
    std::vector<aclmdlDesc*> _modelDescs;
    std::vector<size_t> _numInputNodes;
    std::vector<size_t> _numOutputNodes;

    std::vector<std::vector<const char*>> _modelInputNodeNames;
    std::vector<std::vector<const char*>> _modelOutputNodeNames;
    std::vector<std::vector<std::vector<int64_t>>> _modelInputNodeDims;
    std::vector<std::vector<std::vector<int64_t>>> _modelOutputNodeDims;

    int _imageSize = 512;
    int _videoWidth = 0;
    int _videoHeight = 0;

    //
    std::vector<float> _visionPosEmbeds;
    std::vector<float> _maskMemPosEnc;
    std::vector<float> _maskMemTposEnc;

    // Memory bank
    std::unordered_map<int, MemoryBankEntry> _memoryBank;

    // samurai parameters
    KalmanFilter _kf;
    Eigen::VectorXf _kfMean;
    Eigen::MatrixXf _kfCovariance;
    int _stableFrames = 0;

    int _stableFrameCount = 0;
    float _stableFramesThreshold = 15;
    float _stableIousThreshold = 0.3;
    float _kfScoreWeight = 0.25;
    float _memoryBankIouThreshold = 0.5;
    float _memoryBankObjScoreThreshold = 0.0;
    float _memoryBankKfScoreThreshold = 0.0;
    int _maxObjPtrsInEncoder = 16;
    int _numMaskmem = 7;
}; // class SAM2TrackerAcl

#endif // SAM2_TRACKER_ACL_H
