from pyspark.ml.evaluation import RegressionEvaluator

def regres_evaluator(predictions):
    evaluator = RegressionEvaluator()
    ra2 = evaluator.evaluate(predictions,{evaluator.metricName: "r2"})
    mse = evaluator.evaluate(predictions,{evaluator.metricName: "mse"})
    rmse = evaluator.evaluate(predictions,{evaluator.metricName: "rmse"})
    mae = evaluator.evaluate(predictions,{evaluator.metricName: "mae"})
    print(f"ra2: {ra2}")
    print(f"mse: {mse}")
    print(f"rmse: {rmse}")
    print(f"mae: {mae}")

