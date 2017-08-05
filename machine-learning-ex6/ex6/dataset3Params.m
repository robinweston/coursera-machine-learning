function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

function [bestC, bestSigma] = findBestParams()
    possibleValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    [p,q] = meshgrid(possibleValues, possibleValues);
    combinations = [p(:) q(:)];

    for i=1:length(combinations)
        testC = combinations(i,1);
        testSigma = combinations(i,2);
        model = svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testSigma));
        predictions = svmPredict(model,Xval);
        err = mean(double(predictions ~= yval));
        combinations(i,3) = err;
    end

    [_, minRowIndex] = min(combinations);
    minRow = combinations(minRowIndex(3),:);
    bestC = minRow(1);
    bestSigma = minRow(2);
end

% [C, sigma] = findBestParams();
C = 1;
sigma = 0.1;

% =========================================================================

end
