import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

# Define the input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create the BentoService
model_ref = bentoml.sklearn.get("iris_rf_model:latest")
model_runner = model_ref.to_runner()

# Create a service instance
svc = bentoml.Service("iris_classifier", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=IrisInput), output=JSON())
def predict(input_data: IrisInput):
    data = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    prediction = model_runner.run(data)
    return {"prediction": prediction[0]}
