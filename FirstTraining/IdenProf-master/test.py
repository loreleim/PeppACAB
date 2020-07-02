
from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(execution_path, "idenprof_061-0.7933.h5"))
prediction.setJsonPath(os.path.join(
    execution_path, "idenprof_model_class.json"))
prediction.loadModel(num_objects=10)

predictions, probabilities = prediction.predictImage(
    "hose.jpg", result_count=3)


for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
