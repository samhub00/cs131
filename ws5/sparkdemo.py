import sys

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler

input_path = sys.argv[1]
#output_path = sys.argv[2]

spark = SparkSession.builder.appName("ws5-regression").getOrCreate() #a1

df = (spark.read.format("csv")
	.option("inferSchema", "true")
	.option("header", "true")
	.load(input_path)) #a2


vecAssembler = VectorAssembler(inputCols=["total_bill", "size"], outputCol="features") #a3

trainDF, testDF = df.randomSplit([0.8, 0.2], seed=42) #a4

vecTrainDF = vecAssembler.transform(trainDF)
vecTestDF = vecAssembler.transform(testDF)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

lr = LinearRegression(featuresCol="features", labelCol="tip")
pipeline = Pipeline(stages=[vecAssembler, lr])
pipeline_model = pipeline.fit(trainDF)

predictions = pipeline_model.transform(testDF)

evaluator = RegressionEvaluator(labelCol="tip", predictionCol="prediction")

rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

lr_model = pipeline_model.stages[-1]

print("model performance")
print(f"coefficients: {lr_model.coefficients}")
print(f"Intercept:    {lr_model.intercept}")
print(f"RMSE:         {rmse:.4f}")
print(f"R²:           {r2:.4f}")

spark.stop()


