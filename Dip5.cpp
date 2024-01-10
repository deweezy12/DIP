std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float fractionalMinWeight, float minIsotropy)
{
    // Step 1: Calculate directional gradients
    cv::Mat_<float> gradX, gradY;
    calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);

    // Step 2: Calculate structure tensors
    cv::Mat_<float> A00, A01, A11;
    calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);

    // Step 3: Calculate Foerstner weights and isotropy
    cv::Mat_<float> weight, isotropy;
    calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);

    // Step 4: Find interest points based on the thresholds
    std::vector<cv::Vec2i> interestPoints;

    // Compute the mean of all locally maximal weights
    float meanWeight = 0;
    int count = 0;
    for (int i = 0; i < weight.rows; ++i) {
        for (int j = 0; j < weight.cols; ++j) {
            if (isLocalMaximum(weight, j, i)) {
                meanWeight += weight(i, j);
                ++count;
            }
        }
    }

    meanWeight /= count;

    // Find interest points based on thresholds
    for (int i = 0; i < weight.rows; ++i) {
        for (int j = 0; j < weight.cols; ++j) {
            if (isLocalMaximum(weight, j, i) && weight(i, j) > fractionalMinWeight * meanWeight && isotropy(i, j) > minIsotropy) {
                interestPoints.push_back(cv::Vec2i(j, i)); // Note: x and y are swapped in OpenCV
            }
        }
    }

    return interestPoints;
}
