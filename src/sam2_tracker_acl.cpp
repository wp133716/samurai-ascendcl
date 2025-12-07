#include "sam2_tracker_acl.h"

/**
 * @brief Constructor for SAM2TrackerAcl class.
 * Initializes the object and loads the model from the specified path.
 * 
 * @param modelPath Path to the directory containing the model files.
 */
SAM2TrackerAcl::SAM2TrackerAcl(const std::string &modelPath)
{
    loadNetwork(modelPath);
}

/**
 * @brief Destroy the SAM2TrackerAcl::SAM2TrackerAcl object
 * 
 */
SAM2TrackerAcl::~SAM2TrackerAcl()
{
    unloadModel();
    finalizeACL();
    std::cout << "[SAM2TrackerAcl] ~SAM2TrackerAcl done " << std::endl;
}

/**
 * @brief Initializes the AscendCL environment.
 * This function sets up the AscendCL runtime and checks the current run mode (device or host).
 */
void SAM2TrackerAcl::initAscendCL()
{
    ACL_CHECK(aclInit(nullptr));

    aclrtRunMode runMode;
    ACL_CHECK(aclrtGetRunMode(&runMode));

    switch (runMode)
    {
    case ACL_DEVICE:
        std::cout << "acl get run mode is ACL_DEVICE. " << std::endl;
        break;
    case ACL_HOST:
        std::cout << "acl get run mode is ACL_HOST. " << std::endl;
        break;
    default:
        break;
    }
}

/**
 * @brief Loads the neural network models from the specified path.
 * This function initializes the AscendCL environment, sets the device ID, and loads the models.
 * It also retrieves input/output node information and checks the input size compatibility.
 * 
 * @param modelPath Path to the directory containing the model files.
 */
void SAM2TrackerAcl::loadNetwork(const std::string &modelPath)
{
    // Initialize AscendCL environment
    initAscendCL();

    // Set device ID
    int32_t _deviceId = 0;
    ACL_CHECK(aclrtSetDevice(_deviceId));

    // Number of models
    int numModels = 4;

    // Check if the model path exists
    if (access(modelPath.c_str(), F_OK) == -1)
    {
        throw std::runtime_error("Model path does not exist: " + modelPath);
    }
    std::vector<std::string> modelFiles(numModels);
    modelFiles[0] = modelPath + "/image_encoder.om";
    modelFiles[1] = modelPath + "/memory_attention.om";
    modelFiles[2] = modelPath + "/mask_decoder.om";
    modelFiles[3] = modelPath + "/memory_encoder.om";

    for (int i = 0; i < modelFiles.size(); i++)
    {
        if (i != 1)
        {
            ACL_CHECK(aclrtSetDevice(0)); // memory_attention use _deviceId 1
        }
        else
        {
            ACL_CHECK(aclrtSetDevice(1));
        }

        // Load model file
        ACL_CHECK(aclmdlLoadFromFile(modelFiles[i].c_str(), &(_modelIds[i])));
        std::cout << "\033[33mloading model : " << modelFiles[i] << ", modelId : " << _modelIds[i] << "\033[0m" << std::endl;

        // Create model descriptor
        _modelDescs.push_back(aclmdlCreateDesc());
        ACL_CHECK(aclmdlGetDesc(_modelDescs[i], _modelIds[i]));

        // Get the number of input nodes for the current model
        _numInputNodes.push_back(aclmdlGetNumInputs(_modelDescs[i]));
        std::cout << " inputNum : " << _numInputNodes[i] << std::endl;

        // Get input node names and shapes for the current model
        std::vector<const char *> inputNodeNames;
        std::vector<std::vector<int64_t>> inputNodeDims;
        for (size_t j = 0; j < _numInputNodes[i]; ++j)
        {
            // Get input node name
            inputNodeNames.push_back(aclmdlGetInputNameByIndex(_modelDescs[i], j));
            std::cout << "  " << inputNodeNames[j] << std::endl;

            // Get input node shape
            aclmdlIODims inputDims;
            ACL_CHECK(aclmdlGetInputDims(_modelDescs[i], j, &inputDims));
            std::vector<int64_t> shape(inputDims.dims, inputDims.dims + inputDims.dimCount);
            inputNodeDims.push_back(shape);

            // Print input node shape
            std::cout << "      Shape: ";
            for (const auto &dim : shape)
            {
                std::cout << dim << " ";
            }
            // Print input node data type
            aclDataType dataType = aclmdlGetInputDataType(_modelDescs[i], j);
            printDataType(dataType);
        }
        _modelInputNodeNames.push_back(inputNodeNames);
        _modelInputNodeDims.push_back(inputNodeDims);

        // Get the number of output nodes for the current model
        _numOutputNodes.push_back(aclmdlGetNumOutputs(_modelDescs[i]));
        std::cout << " outputNum : " << _numOutputNodes[i] << std::endl;

        // Get output node names and shapes for the current model
        std::vector<const char *> outputNodeNames;
        std::vector<std::vector<int64_t>> outputNodeDims;
        for (size_t j = 0; j < _numOutputNodes[i]; ++j)
        {
            // Get output node name
            outputNodeNames.push_back(aclmdlGetOutputNameByIndex(_modelDescs[i], j));
            std::cout << "  " << outputNodeNames[j] << std::endl;

            // Get output node shape
            aclmdlIODims outputDims;
            ACL_CHECK(aclmdlGetOutputDims(_modelDescs[i], j, &outputDims));
            std::vector<int64_t> shape(outputDims.dims, outputDims.dims + outputDims.dimCount);
            outputNodeDims.push_back(shape);

            // Print output node shape
            std::cout << "      Shape: ";
            for (const auto &dim : shape)
            {
                std::cout << dim << " ";
            }
            // Print output node data type
            aclDataType dataType = aclmdlGetOutputDataType(_modelDescs[i], j);
            printDataType(dataType);
        }
        _modelOutputNodeNames.push_back(outputNodeNames);
        _modelOutputNodeDims.push_back(outputNodeDims);
    }
    if (_modelInputNodeDims[0][0][1] != _imageSize)
    {
        std::cerr << "_imageSize: " << _imageSize << ", image_encoder input size: " << _modelInputNodeDims[0][0][1] << std::endl;
        throw std::runtime_error("image_encoder input size should be equal to _imageSize");
    }

    //
    _visionPosEmbeds = readBinaryFile(modelPath + "/vision_pos_embeds.bin"); // (1024, 1, 256)
    _maskMemPosEnc = readBinaryFile(modelPath + "/maskmem_pos_enc.bin");     // (1024, 1, 64)
    _maskMemTposEnc = readBinaryFile(modelPath + "/maskmem_tpos_enc.bin");   // (7, 1, 1, 64)
}

/**
 * @brief Prints the data type of a given ACL data type.
 * This function is used for debugging and logging purposes.
 * 
 * @param type The ACL data type to be printed.
 */
void SAM2TrackerAcl::printDataType(aclDataType type)
{
    switch (type)
    {
    case ACL_FLOAT:
        std::cout << "float" << std::endl;
        break;
    case ACL_FLOAT16:
        std::cout << "float16" << std::endl;
        break;
    case ACL_DOUBLE:
        std::cout << "double" << std::endl;
        break;
    case ACL_UINT8:
        std::cout << "uint8" << std::endl;
        break;
    case ACL_INT8:
        std::cout << "int8" << std::endl;
        break;
    case ACL_INT32:
        std::cout << "int32" << std::endl;
        break;
    case ACL_INT64:
        std::cout << "int64" << std::endl;
        break;
    case ACL_BOOL:
        std::cout << "bool" << std::endl;
        break;
    default:
        std::cout << "aclDataType : " << type << std::endl;
        break;
    }
}

/**
 * @brief Performs inference using the image encoder model.
 * This function preprocesses the input frame, executes the image encoder model,
 * and retrieves the output data.
 * 
 * @param frame The input frame as a vector of floats.
 * @param imageEncoderOutput The output data from the image encoder model.
 */
void SAM2TrackerAcl::imageEncoderInference(const std::vector<float> &frame, std::vector<std::vector<float>> &imageEncoderOutput)
{
    auto start = std::chrono::high_resolution_clock::now();

    ACL_CHECK(aclrtSetDevice(0)); // 每次推理前确保设置device

    // Create input dataset
    aclmdlDataset *inputDataset = aclmdlCreateDataset(); // Create an input dataset to hold the input buffers for model inference

    // Get the size of the input buffer for the first input of the model
    size_t bufferSize = aclmdlGetInputSizeByIndex(_modelDescs[0], 0);

    // Copy data from host to device, and adds it to the dataset
    addDataset(frame.data(), bufferSize, inputDataset);

    // Create output dataset
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    createModelOutput(0, outputDataset);

    // Execute the model inference
    aclmdlExecute(_modelIds[0], inputDataset, outputDataset);

    // Retrieve output data
    retrieveOutputData(outputDataset, _numOutputNodes[0], imageEncoderOutput);

    releaseResources(inputDataset, 1); // imageEncoder的输入只有一个，所以为1
    releaseResources(outputDataset, _numOutputNodes[0]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "image_encoder spent: " << duration.count() << " ms" << std::endl;
}

/**
 * @brief Performs inference using the memory attention model.
 * This function processes the vision features and memory bank data,
 * executes the memory attention model, and retrieves the output data.
 * 
 * @param frameIdx The current frame index.
 * @param visionFeatures The vision features extracted from the image encoder.
 * @param memoryAttentionOutputs The output data from the memory attention model.
 */
void SAM2TrackerAcl::memoryAttentionInference(int frameIdx,
                                              const std::vector<float> &visionFeatures,
                                              std::vector<std::vector<float>> &memoryAttentionOutputs)
{
    auto start1 = std::chrono::high_resolution_clock::now();
    std::vector<float> memmaskFeatures = _memoryBank[0].maskmem_features;
    std::vector<float> memmaskPosEncs = _maskMemPosEnc;
    std::vector<float> objectPtrs = _memoryBank[0].obj_ptr;
    // // std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() << std::endl;
    // // std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size() << std::endl;
    // // std::cout << "objectPtrs.size(): " << objectPtrs.size() << std::endl;

    // std::vector<int> validIndices;
    // if (frameIdx > 1)
    // {
    //     for (int i = frameIdx - 1; i > 0; i--)
    //     {
    //         float iouScore = _memoryBank[i].best_iou_score;
    //         float objScore = _memoryBank[i].obj_score_logits;
    //         float kfScore = _memoryBank[i].kf_score;
    //         // if (iouScore > _memoryBankIouThreshold && objScore > _memoryBankObjScoreThreshold && (kfScore > _memoryBankKfScoreThreshold))
    //         if (1)
    //         {
    //             validIndices.insert(validIndices.begin(), i);
    //         }
    //         if (validIndices.size() >= _maxObjPtrsInEncoder - 1)
    //         {
    //             break;
    //         }
    //     }
    // }
    // // std::cout << "validIndices.size(): " << validIndices.size() << std::endl;
    // // std::cout << "validIndices : ";
    // // for (int i = 0; i < validIndices.size(); i++) {
    // //     std::cout << validIndices[i] << ", ";
    // // }
    // // std::cout << std::endl;

    // size_t maskmemFeaturesSize = aclmdlGetOutputSizeByIndex(_modelDescs[3], 0) / sizeof(float); // 262144=4096*64 65536=1024*64
    // size_t maskmemPosEncSize = maskmemFeaturesSize;
    // size_t objPtrSize = _modelOutputNodeDims[2][2][2]; // 256
    // // std::cout << "maskmemFeaturesSize: " << maskmemFeaturesSize << std::endl;
    // // std::cout << "maskmemPosEncSize: " << maskmemPosEncSize << std::endl;
    // // std::cout << "objPtrSize: " << objPtrSize << std::endl;
    // size_t memmaskFeaturesNum = std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);
    // size_t memmaskPosEncNum = std::min(static_cast<size_t>(_numMaskmem), validIndices.size() + 1);

    // memmaskFeatures.reserve(maskmemFeaturesSize * memmaskFeaturesNum); // 最近 num_maskmem-1 帧 + 0帧 的maskmem_features
    // memmaskPosEncs.reserve(maskmemPosEncSize * memmaskPosEncNum);      // 最近 num_maskmem-1 帧 + 0帧 的maskmem_pos_enc

    // // std::cout << "memmaskFeatures idx: ";
    // int validIndicesSize = validIndices.size();
    // for (int i = validIndicesSize - _numMaskmem + 1; i < validIndicesSize; i++)
    // {
    //     if (i < 0)
    //     {
    //         continue;
    //     }
    //     // std::cout << i << ": " << validIndices[i] << ", ";

    //     int prevFrameIdx = validIndices[i];
    //     MemoryBankEntry mem = _memoryBank[prevFrameIdx];
    //     memmaskFeatures.insert(memmaskFeatures.end(), mem.maskmem_features.begin(), mem.maskmem_features.end());
    //     memmaskPosEncs.insert(memmaskPosEncs.end(), mem.maskmem_pos_enc.begin(), mem.maskmem_pos_enc.end());
    // }
    // // std::cout << std::endl;

    // auto start2 = std::chrono::high_resolution_clock::now();
    // // std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
    // std::vector<int64_t> tposEncSize = {7, 1, 1, 64};

    // for (int i = 1; i < memmaskFeaturesNum; i++)
    // {
    //     int start = (memmaskFeaturesNum - i) * maskmemPosEncSize;
    //     int end = start + maskmemPosEncSize;
    //     // #pragma omp parallel for
    //     for (int j = start; j < end; j++)
    //     {
    //         memmaskPosEncs[j] += _maskMemTposEnc[(i - 1) * tposEncSize[3] + (j % tposEncSize[3])];
    //     }
    // }
    // // #pragma omp parallel for
    // for (int i = 0; i < maskmemFeaturesSize; i++)
    // {
    //     memmaskPosEncs[i] += _maskMemTposEnc[(tposEncSize[0] - 1) * tposEncSize[3] + (i % tposEncSize[3])];
    // }
    // auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2);
    // std::cout << "memmaskPosEncs spent: " << duration2.count() << " ms" << std::endl;

    std::vector<int> objPosEnc = {frameIdx};
    // // std::cout << "objPosEnc : ";
    // for (int i = 1; i < frameIdx; i++)
    // {
    //     // std::cout << i << ", ";
    //     objPosEnc.push_back(i);
    //     if (objPosEnc.size() >= _maxObjPtrsInEncoder)
    //     {
    //         break;
    //     }
    // }
    // // std::cout << std::endl;

    // objectPtrs.reserve(objPtrSize * objPosEnc.size()); // 最近 maxObjPtrsInEncoder-1 帧 + 0帧 的obj_ptr
    // // std::cout << "objectPtrs : ";
    // for (int i = frameIdx - 1; i > 0; i--)
    // {
    //     // std::cout << i << ", ";
    //     MemoryBankEntry mem = _memoryBank[i];
    //     objectPtrs.insert(objectPtrs.end(), mem.obj_ptr.begin(), mem.obj_ptr.end());
    //     if (objectPtrs.size() >= _maxObjPtrsInEncoder * objPtrSize)
    //     {
    //         break;
    //     }
    // }
    // // std::cout << std::endl;

    // std::cout << "memmaskFeaturesNum: " << memmaskFeaturesNum << std::endl;
    // std::cout << "memmaskPosEncNum: " << memmaskPosEncNum << std::endl;
    // std::cout << "validIndices.size(): " << validIndices.size() << std::endl;
    // // std::cout << "objectPtrs.size(): " << objectPtrs.size() / objPtrSize << std::endl;
    // std::cout << "objPosEnc.size(): " << objPosEnc.size() << std::endl;
    // std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() / maskmemFeaturesSize << ", " << memmaskFeatures.capacity() / maskmemFeaturesSize << std::endl;
    // std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size() / maskmemPosEncSize<< ", " << memmaskPosEncs.capacity() / maskmemPosEncSize << std::endl;

    ACL_CHECK(aclrtSetDevice(_deviceId)); // 每次推理前确保设置device

    // Create input dataset
    aclmdlDataset *inputDataset = aclmdlCreateDataset(); // Create an input dataset to hold the input buffers for model inference

    // Copy data from host to device, and adds it to the dataset
    addDataset(visionFeatures.data(),   aclmdlGetInputSizeByIndex(_modelDescs[1], 0), inputDataset);
    addDataset(_visionPosEmbeds.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 1), inputDataset);

    addDataset(memmaskFeatures.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 2), inputDataset);

    // std::cout << "memmaskPosEncs begin " << std::endl;
    addDataset(memmaskPosEncs.data(),  aclmdlGetInputSizeByIndex(_modelDescs[1], 3), inputDataset);

    // bool hasAbnormalData = std::any_of(memmaskPosEncs.begin(), memmaskPosEncs.end(), isAbnormal);
    // if (hasAbnormalData) {
    //     std::cout << "Vector contains abnormal data!" << std::endl;
    // } else {
    //     // std::cout << "Vector is normal." << std::endl;
    // }
    // std::cout << "memmaskPosEncs end " << std::endl;

    addDataset(objectPtrs.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 4), inputDataset);
    addDataset(objPosEnc.data(),  aclmdlGetInputSizeByIndex(_modelDescs[1], 5), inputDataset);

    // std::vector<int> ascendMbatchShapeData = {static_cast<int>(memmaskFeaturesNum),
    //                                           static_cast<int>(memmaskFeaturesNum),
    //                                           static_cast<int>(objPosEnc.size()),
    //                                           static_cast<int>(objPosEnc.size())};
    // addDataset(ascendMbatchShapeData.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 6), inputDataset);

    // Create output dataset
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    createModelOutput(1, outputDataset);

    // Execute the model inference
    aclmdlExecute(_modelIds[1], inputDataset, outputDataset);

    // Retrieve output data
    retrieveOutputData(outputDataset, _numOutputNodes[1], memoryAttentionOutputs);

    releaseResources(inputDataset, _numInputNodes[1] + 1); // 如果是动态输入，则还要加上ascendMbatchShapeData，所以总的输入个数为_numInputNodes[1] + 1
    releaseResources(outputDataset, _numOutputNodes[1]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1);
    std::cout << "memory_attention spent: " << duration.count() << " ms" << std::endl;
}

/**
 * @brief Performs inference using the mask decoder model.
 * This function processes the input points, labels, and features,
 * executes the mask decoder model, and retrieves the output data.
 * 
 * @param inputPoints The input points for the mask decoder.
 * @param inputLabels The input labels for the mask decoder.
 * @param pixFeatWithMem The pixel features with memory from the memory attention model.
 * @param highResFeatures0 The high-resolution features from the image encoder.
 * @param highResFeatures1 The high-resolution features from the image encoder.
 * @param maskDecoderOutputs The output data from the mask decoder model.
 */
void SAM2TrackerAcl::maskDecoderInference(const std::vector<float> &inputPoints,
                                          const std::vector<int>   &inputLabels,
                                          const std::vector<float> &pixFeatWithMem,
                                          const std::vector<float> &highResFeatures0,
                                          const std::vector<float> &highResFeatures1,
                                          std::vector<std::vector<float>> &maskDecoderOutputs)
{
    auto start = std::chrono::high_resolution_clock::now();

    ACL_CHECK(aclrtSetDevice(0));

    // Create input dataset
    aclmdlDataset *inputDataset = aclmdlCreateDataset(); // Create an input dataset to hold the input buffers for model inference

    // Copy data from host to device, and adds it to the dataset
    addDataset(inputPoints.data(), aclmdlGetInputSizeByIndex(_modelDescs[2], 0), inputDataset);
    addDataset(inputLabels.data(), aclmdlGetInputSizeByIndex(_modelDescs[2], 1), inputDataset);

    addDataset(pixFeatWithMem.data(), aclmdlGetInputSizeByIndex(_modelDescs[2], 2), inputDataset);

    addDataset(highResFeatures0.data(), aclmdlGetInputSizeByIndex(_modelDescs[2], 3), inputDataset);
    addDataset(highResFeatures1.data(), aclmdlGetInputSizeByIndex(_modelDescs[2], 4), inputDataset);

    // Create output dataset
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    createModelOutput(2, outputDataset);

    // Execute the model inference
    aclmdlExecute(_modelIds[2], inputDataset, outputDataset);

    // Retrieve output data
    retrieveOutputData(outputDataset, _numOutputNodes[2], maskDecoderOutputs);

    releaseResources(inputDataset, _numInputNodes[2]);
    releaseResources(outputDataset, _numOutputNodes[2]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "mask_decoder spent: " << duration.count() << " ms" << std::endl;
}

/**
 * @brief Performs inference using the memory encoder model.
 * This function processes the vision features, high-resolution masks, and object scores,
 * executes the memory encoder model, and retrieves the output data.
 * 
 * @param visionFeatures The vision features extracted from the image encoder.
 * @param highResMasksForMem The high-resolution masks for the mask decoder.
 * @param objectScoreLogits The object score logits.
 * @param isMaskFromPts A flag indicating whether the mask is generated from points.
 * @param memoryEncoderOutputs The output data from the memory encoder model.
 */
void SAM2TrackerAcl::memoryEncoderInference(const std::vector<float> &visionFeatures,
                                            const std::vector<float> &highResMasksForMem,
                                            const std::vector<float> &objectScoreLogits,
                                            bool isMaskFromPts,
                                            std::vector<std::vector<float>> &memoryEncoderOutputs)
{
    auto start = std::chrono::high_resolution_clock::now();
    ACL_CHECK(aclrtSetDevice(0));

    // Create input dataset
    aclmdlDataset *inputDataset = aclmdlCreateDataset(); // Create an input dataset to hold the input buffers for model inference

    // Copy data from host to device, and adds it to the dataset
    addDataset(visionFeatures.data(),     aclmdlGetInputSizeByIndex(_modelDescs[3], 0), inputDataset);
    addDataset(highResMasksForMem.data(), aclmdlGetInputSizeByIndex(_modelDescs[3], 1), inputDataset);
    addDataset(objectScoreLogits.data(),  aclmdlGetInputSizeByIndex(_modelDescs[3], 2), inputDataset);
    addDataset(&isMaskFromPts,            aclmdlGetInputSizeByIndex(_modelDescs[3], 3), inputDataset);

    // Create output dataset
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    createModelOutput(3, outputDataset);

    // Execute the model inference
    aclmdlExecute(_modelIds[3], inputDataset, outputDataset);

    // Retrieve output data
    retrieveOutputData(outputDataset, _numOutputNodes[3], memoryEncoderOutputs);

    releaseResources(inputDataset, _numInputNodes[3]);
    releaseResources(outputDataset, _numOutputNodes[3]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "memory_encoder spent: " << duration.count() << " ms" << std::endl;
}

/**
 * @brief Adds the first frame bounding box to the memory bank.
 * This function processes the first frame, performs inference using the image encoder and mask decoder,
 * and stores the results in the memory bank.
 * 
 * @param frameIdx The current frame index.
 * @param firstFrame The first frame of the video.
 * @param bbox The bounding box for the first frame.
 * @return cv::Mat The predicted mask for the first frame.
 */
cv::Mat SAM2TrackerAcl::addFirstFrameBbox(int frameIdx, const cv::Mat &firstFrame, const cv::Rect &bbox)
{
    _videoWidth = static_cast<int>(firstFrame.cols);
    _videoHeight = static_cast<int>(firstFrame.rows);

    std::vector<float> inputImage;
    preprocessImage(firstFrame, inputImage);

    // 1) image_encoder 推理
    std::vector<std::vector<float>> imageEncoderOutputs;
    imageEncoderInference(inputImage, imageEncoderOutputs);

    // 2) mask_decoder 推理
    std::vector<float> inputPoints = {static_cast<float>(bbox.x), static_cast<float>(bbox.y),
                                      static_cast<float>(bbox.x + bbox.width), static_cast<float>(bbox.y + bbox.height)};
    inputPoints[0] = (inputPoints[0] / firstFrame.cols) * _imageSize;
    inputPoints[1] = (inputPoints[1] / firstFrame.rows) * _imageSize;
    inputPoints[2] = (inputPoints[2] / firstFrame.cols) * _imageSize;
    inputPoints[3] = (inputPoints[3] / firstFrame.rows) * _imageSize;

    std::vector<int> boxLabels = {2, 3};

    std::vector<std::vector<float>> maskDecoderOutputs;
    maskDecoderInference(inputPoints, boxLabels,
                         imageEncoderOutputs[3], // pixFeatWithMem
                         imageEncoderOutputs[0], // highResFeatures0
                         imageEncoderOutputs[1], // highResFeatures1
                         maskDecoderOutputs);

    std::vector<float> lowResMultiMasks = maskDecoderOutputs[0];
    std::vector<float> ious = maskDecoderOutputs[1];
    std::vector<float> objPtrs = maskDecoderOutputs[2];
    std::vector<float> objScoreLogits = maskDecoderOutputs[3];

    PostprocessResult result = postprocessOutput(maskDecoderOutputs);
    int bestIoUIndex = result.bestIoUIndex;
    float bestIouScore = result.bestIouScore;
    float kfScore = result.kfScore;

    int lowResMaskHeight = _modelOutputNodeDims[2][0][2];
    int lowResMaskWidth = _modelOutputNodeDims[2][0][3];
    auto lowResMask = lowResMultiMasks.data() + bestIoUIndex * lowResMaskWidth * lowResMaskHeight;
    cv::Mat predMask(lowResMaskWidth, lowResMaskHeight, CV_32FC1, lowResMask);

    // 3) memory_encoder 推理
    bool isMaskFromPts = frameIdx == 0;

    cv::Mat highResMask;
    cv::resize(predMask, highResMask, cv::Size(_imageSize, _imageSize));

    std::vector<float> highResMaskForMem(highResMask.begin<float>(), highResMask.end<float>());

    std::vector<std::vector<float>> memoryEncoderOutputs;
    memoryEncoderInference(imageEncoderOutputs[2], // visionFeatures
                           highResMaskForMem,
                           objScoreLogits,
                           isMaskFromPts,
                           memoryEncoderOutputs);

    std::vector<float> maskmemFeatures = memoryEncoderOutputs[0];

    // 4) save memory bank
    int objPtrSize = _modelOutputNodeDims[2][2][2]; // 256 DecoderOutputNodeDims[2][2] 1,3,256

    MemoryBankEntry entry;
    entry.maskmem_features = maskmemFeatures;
    entry.maskmem_pos_enc = _maskMemPosEnc; // _maskMemPosEnc
    entry.obj_ptr = std::vector<float>(objPtrs.data() + bestIoUIndex * objPtrSize, objPtrs.data() + (bestIoUIndex + 1) * objPtrSize);
    entry.best_iou_score = bestIouScore;
    entry.obj_score_logits = objScoreLogits[0];
    entry.kf_score = kfScore;

    _memoryBank[frameIdx] = entry;

    return predMask;
}

/**
 * @brief Performs tracking for a given frame.
 * This function processes the input frame, performs inference using the image encoder, memory attention,
 * and mask decoder models, and updates the memory bank.
 * 
 * @param frameIdx The current frame index.
 * @param frame The input frame.
 * @return cv::Mat The predicted mask for the current frame.
 */
cv::Mat SAM2TrackerAcl::trackStep(int frameIdx, const cv::Mat &frame)
{
    std::vector<float> inputImage;
    preprocessImage(frame, inputImage);

    // 1) image_encoder 推理
    std::vector<std::vector<float>> imageEncoderOutputs;
    imageEncoderInference(inputImage, imageEncoderOutputs);
    // std::vector<float> lowResFeatures = imageEncoderOutputs[2], // visionFeatures

    // 2) memory_attention 推理
    std::vector<std::vector<float>> memoryAttentionOutputs;
    memoryAttentionInference(frameIdx, imageEncoderOutputs[2], memoryAttentionOutputs);

    // 3) mask_decoder 推理
    std::vector<float> inputPoints = {0, 0, 0, 0};
    std::vector<int> inputLabels = {-1, -1};

    std::vector<std::vector<float>> maskDecoderOutputs;
    maskDecoderInference(inputPoints, inputLabels,
                         memoryAttentionOutputs[0], // pixFeatWithMem
                         imageEncoderOutputs[0],    // highResFeatures0
                         imageEncoderOutputs[1],    // highResFeatures1
                         maskDecoderOutputs);

    std::vector<float> lowResMultiMasks = maskDecoderOutputs[0];
    std::vector<float> ious = maskDecoderOutputs[1];
    std::vector<float> objPtrs = maskDecoderOutputs[2];
    std::vector<float> objScoreLogits = maskDecoderOutputs[3];

    PostprocessResult result = postprocessOutput(maskDecoderOutputs);
    int bestIoUIndex = result.bestIoUIndex;
    float bestIouScore = result.bestIouScore;
    float kfScore = result.kfScore;

    int lowResMaskHeight = _modelOutputNodeDims[2][0][2];
    int lowResMaskWidth = _modelOutputNodeDims[2][0][3];
    auto lowResMask = lowResMultiMasks.data() + bestIoUIndex * lowResMaskWidth * lowResMaskHeight;
    cv::Mat predMask(lowResMaskWidth, lowResMaskHeight, CV_32FC1, lowResMask);

    // 4) memory_encoder 推理
    // bool isMaskFromPts = frameIdx == 0;

    // cv::Mat highResMask;
    // cv::resize(predMask, highResMask, cv::Size(_imageSize, _imageSize));

    // // std::vector<float> highResMaskForMem((float*)highResMask.data, (float*)highResMask.data + highResMask.total());
    // std::vector<float> highResMaskForMem(highResMask.begin<float>(), highResMask.end<float>());

    // std::vector<std::vector<float>> memoryEncoderOutputs;
    // memoryEncoderInference(imageEncoderOutputs[2], // visionFeatures
    //                        highResMaskForMem,
    //                        objScoreLogits,
    //                        isMaskFromPts,
    //                        memoryEncoderOutputs);

    // std::vector<float> maskmemFeatures = memoryEncoderOutputs[0];

    // 5) save memory bank
    // int objPtrSize = _modelOutputNodeDims[2][2][2]; // 256 DecoderOutputNodeDims[2][2] 1,3,256

    // MemoryBankEntry entry;
    // entry.maskmem_features = maskmemFeatures;
    // entry.maskmem_pos_enc = _maskMemPosEnc; // _maskMemPosEnc
    // entry.obj_ptr = std::vector<float>(objPtrs.data() + bestIoUIndex * objPtrSize, objPtrs.data() + (bestIoUIndex + 1) * objPtrSize);
    // entry.best_iou_score = bestIouScore;
    // entry.obj_score_logits = objScoreLogits[0];
    // entry.kf_score = kfScore;

    // //
    // if (_memoryBank.size() >= _maxObjPtrsInEncoder)
    // {
    //     int eraseIdx = frameIdx - _maxObjPtrsInEncoder + 1;
    //     _memoryBank.erase(eraseIdx);
    // }
    // _memoryBank[frameIdx] = entry;

    return predMask;
}

/**
 * @brief Preprocesses the input image.
 * This function resizes the input image, converts it to RGB, and normalizes it.
 * 
 * @param src The input image.
 * @param dest The preprocessed image as a vector of floats.
 */
void SAM2TrackerAcl::preprocessImage(const cv::Mat &src, std::vector<float> &dest)
{
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(_imageSize, _imageSize));
    cv::Mat rgbImage;
    cv::cvtColor(resized, rgbImage, cv::COLOR_BGR2RGB); // 转换为RGB
    rgbImage.convertTo(rgbImage, CV_32FC3);             // 转换为float
    dest.assign((float *)rgbImage.data, (float *)rgbImage.data + rgbImage.total() * rgbImage.channels());

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "preprocessImage spent: " << duration.count() << " ms" << std::endl;
}

/**
 * @brief Postprocesses the output of the mask decoder.
 * This function selects the best mask based on IoU scores and Kalman filter predictions.
 * 
 * @param maskDecoderOutputs The output data from the mask decoder.
 * @return PostprocessResult The postprocessed result containing the best IoU index, score, and Kalman filter score.
 */
PostprocessResult SAM2TrackerAcl::postprocessOutput(const std::vector<std::vector<float>> &maskDecoderOutputs)
{
    // maskDecoderOutputs[0] : lowResMultiMasks，(3, videoW, videoH)
    // maskDecoderOutputs[1] : ious, (3)
    // maskDecoderOutputs[2] : objPtrs, (1, 3, 256)
    // maskDecoderOutputs[3] : objScoreLogits, (1, 1)
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> lowResMultiMasks = maskDecoderOutputs[0];
    std::vector<float> ious = maskDecoderOutputs[1];
    std::vector<float> objPtrs = maskDecoderOutputs[2];
    std::vector<float> objScoreLogits = maskDecoderOutputs[3];

    int numMasks = ious.size();

#if 0 // sam2: 选择ious最高的index
    int bestIoUIndex = std::distance(ious.data(), std::max_element(ious.data(), ious.data() + numMasks));
    float bestIouScore = ious[bestIoUIndex];
    float kfScore = 1.0;

#else // samurai: 加入卡尔曼滤波预测
    int bestIoUIndex;
    float bestIouScore;
    float kfScore = 1.0;

    if ((_kfMean.size() == 0 && _kfCovariance.size() == 0) || _stableFrameCount == 0)
    {
        bestIoUIndex = std::distance(ious.data(), std::max_element(ious.data(), ious.data() + numMasks));
        bestIouScore = ious[bestIoUIndex];

        int lowResMaskSize = _modelOutputNodeDims[2][0][2];
        float *lowResMask = lowResMultiMasks.data() + bestIoUIndex * lowResMaskSize * lowResMaskSize;
        cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

        cv::Rect bbox(0, 0, 0, 0);
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(binaryMask, nonZeroPoints);
        if (!nonZeroPoints.empty())
        {
            bbox = cv::boundingRect(nonZeroPoints);
        }

        // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]" << std::endl;

        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.initiate(
                                                                    _kf.xyxy2xyah(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)));
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        _stableFrameCount++;
    }
    else if (_stableFrameCount < _stableFramesThreshold)
    {
        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.predict(_kfMean, _kfCovariance);
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        bestIoUIndex = std::distance(ious.data(), std::max_element(ious.data(), ious.data() + numMasks));
        bestIouScore = ious[bestIoUIndex];

        int lowResMaskSize = _modelOutputNodeDims[2][0][2];
        float *lowResMask = lowResMultiMasks.data() + bestIoUIndex * lowResMaskSize * lowResMaskSize;
        cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

        cv::Mat binaryMask;
        cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

        cv::Rect bbox(0, 0, 0, 0);
        std::vector<cv::Point> nonZeroPoints;
        cv::findNonZero(binaryMask, nonZeroPoints);
        if (!nonZeroPoints.empty())
        {
            bbox = cv::boundingRect(nonZeroPoints);
        }

        // std::cout << "bbox: [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]" << std::endl;

        if (bestIouScore > _stableIousThreshold)
        {
            std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(_kfMean, _kfCovariance,
                                                                              _kf.xyxy2xyah(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)));
            _kfMean = kfResult.first;
            _kfCovariance = kfResult.second;
            _stableFrameCount++;
        }
        else
        {
            _stableFrameCount = 0;
        }
    }
    else
    {
        std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.predict(_kfMean, _kfCovariance);
        _kfMean = kfResult.first;
        _kfCovariance = kfResult.second;

        std::vector<Eigen::Vector4f> predBboxs;
        for (int i = 0; i < numMasks; i++)
        {
            int lowResMaskSize = _modelOutputNodeDims[2][0][2];
            float *lowResMask = lowResMultiMasks.data() + i * lowResMaskSize * lowResMaskSize;
            cv::Mat predMask(lowResMaskSize, lowResMaskSize, CV_32FC1, lowResMask);

            cv::Mat binaryMask;
            cv::threshold(predMask, binaryMask, 0.01, 1.0, cv::THRESH_BINARY);

            cv::Rect bbox(0, 0, 0, 0);
            std::vector<cv::Point> nonZeroPoints;
            cv::findNonZero(binaryMask, nonZeroPoints);
            if (!nonZeroPoints.empty())
            {
                bbox = cv::boundingRect(nonZeroPoints);
            }

            predBboxs.push_back(Eigen::Vector4f(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height));
        }

        std::vector<float> kfIousVec = _kf.computeIoUs(_kfMean.head(4), predBboxs);

        std::vector<float> weightedIous;
        for (int i = 0; i < numMasks; i++)
        {
            weightedIous.push_back(_kfScoreWeight * kfIousVec[i] + (1 - _kfScoreWeight) * ious[i]);
        }

        bestIoUIndex = std::distance(weightedIous.begin(), std::max_element(weightedIous.begin(), weightedIous.end()));
        bestIouScore = ious[bestIoUIndex];
        kfScore = kfIousVec[bestIoUIndex];

        if (bestIouScore > _stableIousThreshold)
        {
            std::pair<Eigen::VectorXf, Eigen::MatrixXf> kfResult = _kf.update(_kfMean, _kfCovariance,
                                                                              _kf.xyxy2xyah(predBboxs[bestIoUIndex]));
            _kfMean = kfResult.first;
            _kfCovariance = kfResult.second;
        }
        else
        {
            _stableFrameCount = 0;
        }
    }

#endif

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "postprocess spent: " << duration.count() << " ms" << std::endl;

    return {bestIoUIndex, bestIouScore, kfScore};
}

/**
 * @brief Adds data to the input dataset for model inference.
 * This function allocates device memory, copies data from host to device, and adds it to the dataset.
 * 
 * @param hostData The input data on the host.
 * @param bufferSize The size of the input data buffer.
 * @param dataset The input dataset for model inference.
 */
void SAM2TrackerAcl::addDataset(const void *hostData, size_t bufferSize, aclmdlDataset *dataset)
{
    // Allocate memory on the device (NPU) for the input buffer
    void *devicePtr = nullptr;
    ACL_CHECK(aclrtMalloc(&devicePtr, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST)); // ACL_MEM_MALLOC_NORMAL_ONLY ACL_MEM_MALLOC_HUGE_FIRST
    if (devicePtr == nullptr)
    {
        std::cout << "Failed to allocate memory for input." << std::endl;
    }
    else
    {
        // std::cout << "success to allocate memory for input." << std::endl;
    }

    // Copy data from the host (CPU) memory to the device (NPU) memory
    ACL_CHECK(aclrtMemcpy(devicePtr, bufferSize, hostData, bufferSize, ACL_MEMCPY_DEFAULT)); // ACL_MEMCPY_DEVICE_TO_HOST

    aclDataBuffer *dataBuffer = aclCreateDataBuffer(devicePtr, bufferSize); // Create a data buffer to wrap the device memory containing the input data
    ACL_CHECK(aclmdlAddDatasetBuffer(dataset, dataBuffer));                 // Add the data buffer to the input dataset
}

/**
 * @brief Creates the output dataset for model inference.
 * This function allocates device memory for the output buffers and adds them to the dataset.
 * 
 * @param modelDescIdx The index of the model descriptor.
 * @param outputDataset The output dataset for model inference.
 */
void SAM2TrackerAcl::createModelOutput(int modelDescIdx, aclmdlDataset *outputDataset)
{
    for (size_t i = 0; i < _numOutputNodes[modelDescIdx]; ++i)
    {
        size_t bufferSize = aclmdlGetOutputSizeByIndex(_modelDescs[modelDescIdx], i);

        void *buffer = nullptr;
        ACL_CHECK(aclrtMalloc(&buffer, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (buffer == nullptr)
        {
            std::cout << "Failed to allocate memory for output_" << i << std::endl;
        }
        else
        {
            // std::cout << "success to allocate memory for output_" << i << std::endl;
        }

        aclDataBuffer *dataBuffer = aclCreateDataBuffer(buffer, bufferSize);
        ACL_CHECK(aclmdlAddDatasetBuffer(outputDataset, dataBuffer));
    }
}

/**
 * @brief Retrieves output data from the output dataset.
 * This function copies the output data from device memory to host memory.
 * 
 * @param outputDataset The output dataset from model inference.
 * @param numOutputNodes The number of output nodes.
 * @param outputData The output data as a vector of vectors.
 */
void SAM2TrackerAcl::retrieveOutputData(aclmdlDataset *outputDataset, size_t numOutputNodes, std::vector<std::vector<float>> &outputData)
{
    outputData.clear();

    // Retrieve output data
    for (size_t i = 0; i < numOutputNodes; ++i)
    {
        // Get the output data buffer for the current output node
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(outputDataset, i);
        if (dataBuffer == nullptr)
        {
            std::cerr << "Failed to get data buffer for output " << i << std::endl;
        }

        // Get the pointer to the output data and its size
        void *data = aclGetDataBufferAddr(dataBuffer);        // Device memory address
        size_t dataSize = aclGetDataBufferSizeV2(dataBuffer); // Size of the output data in bytes

        // Copy the output data from device memory to host memory
        std::vector<float> hostData(dataSize / sizeof(float)); // Create a host buffer to store the data
        ACL_CHECK(aclrtMemcpy(hostData.data(), dataSize, data, dataSize, ACL_MEMCPY_DEFAULT));

        outputData.push_back(std::move(hostData));
    }
}

/**
 * @brief Releases resources used for model inference.
 * This function frees device memory and destroys the dataset.
 * 
 * @param dataset The dataset to be released.
 * @param dataSize The size of the data buffers in the dataset.
 */
void SAM2TrackerAcl::releaseResources(aclmdlDataset *dataset, int dataSize)
{
    for (size_t i = 0; i < dataSize; ++i)
    {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        if (dataBuffer != nullptr)
        {
            ACL_CHECK(aclrtFree(aclGetDataBufferAddr(dataBuffer)));
        }
    }

    ACL_CHECK(aclmdlDestroyDataset(dataset));
}

/**
 * @brief Unloads the models and releases associated resources.
 * This function destroys the model descriptors and unloads the models.
 */
void SAM2TrackerAcl::unloadModel()
{
    std::cout << "[SAM2TrackerAcl] unloadModel " << std::endl;

    for (auto &desc : _modelDescs)
    {
        ACL_CHECK(aclmdlDestroyDesc(desc));
    }
    for (auto &mId : _modelIds)
    {
        ACL_CHECK(aclmdlUnload(mId));
    }
}

/**
 * @brief Finalizes the AscendCL environment.
 * This function resets the device and finalizes the AscendCL runtime.
 */
void SAM2TrackerAcl::finalizeACL()
{
    std::cout << "[SAM2TrackerAcl] finalizeACL " << std::endl;

    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclrtResetDevice(1));
    ACL_CHECK(aclFinalize());
}

/**
 * @brief Reads binary data from a file.
 * This function reads the contents of a binary file and returns it as a vector of floats.
 * 
 * @param filePath The path to the binary file.
 * @return std::vector<float> The data read from the file.
 */
std::vector<float> SAM2TrackerAcl::readBinaryFile(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {};
    }

    // Get file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file content into buffer
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
    {
        std::cerr << "Failed to read file: " << filePath << std::endl;
        return {};
    }

    file.close();
    return buffer;
}

void SAM2TrackerAcl::test_memoryAttentionInference(std::vector<std::vector<float>> &memoryAttentionOutputs)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> memmaskFeatures(1024 * 1 * 64);
    std::vector<float> memmaskPosEncs(1024 * 1 * 64);
    std::vector<float> objectPtrs(1 * 1 * 256);
    // // std::cout << "memmaskFeatures.size(): " << memmaskFeatures.size() << std::endl;
    // // std::cout << "memmaskPosEncs.size(): " << memmaskPosEncs.size() << std::endl;
    // // std::cout << "objectPtrs.size(): " << objectPtrs.size() << std::endl;
    std::vector<float> visionFeatures(1024 * 1 * 256);
    // std::vector<float> _visionPosEmbeds(1024 * 1 * 256);
    std::vector<float> objPosEnc(1);

    ACL_CHECK(aclrtSetDevice(1));

    // Create input dataset
    aclmdlDataset *inputDataset = aclmdlCreateDataset(); // Create an input dataset to hold the input buffers for model inference

    // Copy data from host to device, and adds it to the dataset
    addDataset(visionFeatures.data(),   aclmdlGetInputSizeByIndex(_modelDescs[1], 0), inputDataset); // 1024 1 256
    addDataset(_visionPosEmbeds.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 1), inputDataset); // 1024 1 256

    addDataset(memmaskFeatures.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 2), inputDataset); // 1024 1 1 64

    addDataset(memmaskPosEncs.data(),  aclmdlGetInputSizeByIndex(_modelDescs[1], 3), inputDataset); // 1024 1 1 64

    addDataset(objectPtrs.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 4), inputDataset); // 1 1 256
    addDataset(objPosEnc.data(),  aclmdlGetInputSizeByIndex(_modelDescs[1], 5), inputDataset);

    // std::vector<int> ascendMbatchShapeData = {static_cast<int>(memmaskFeaturesNum),
    //                                           static_cast<int>(memmaskFeaturesNum),
    //                                           static_cast<int>(objPosEnc.size()),
    //                                           };
    // addDataset(ascendMbatchShapeData.data(), aclmdlGetInputSizeByIndex(_modelDescs[1], 6), inputDataset);

    // Create output dataset
    aclmdlDataset *outputDataset = aclmdlCreateDataset();
    createModelOutput(1, outputDataset);

    // Execute the model inference
    auto startAclmdlExecute = std::chrono::high_resolution_clock::now();
    aclmdlExecute(_modelIds[1], inputDataset, outputDataset);
    auto durationAclmdlExecute = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startAclmdlExecute);
    std::cout << "aclmdlExecute spent: " << durationAclmdlExecute.count() << " ms" << std::endl;

    // Retrieve output data
    retrieveOutputData(outputDataset, _numOutputNodes[1], memoryAttentionOutputs);

    releaseResources(inputDataset, _numInputNodes[1] + 1);
    releaseResources(outputDataset, _numOutputNodes[1]);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "memory_attention spent: " << duration.count() << " ms" << std::endl;
}
