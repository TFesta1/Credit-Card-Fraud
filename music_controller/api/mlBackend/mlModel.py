import json
modelData = None

def trainModel(): 
    global modelData
    import time
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, count, isnan, when, explode, array, lit, udf
    from pyspark.sql.types import ArrayType, FloatType
    
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import StandardScaler
    from pyspark.ml.feature import StringIndexer

    from pyspark.ml.classification import GBTClassifier

    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.feature import MinMaxScaler

    # time.sleep(4)
    # modelData = ("Model", [[1,2,3,4,5], [2,3,4,5,6]])
    # return

    # Create a Spark session
    spark = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()
    # spark.range(5).show() #Dataframe, column id 0 to 4.
    file_path = r"C:\Users\ringk\OneDrive\Documents\CreditCard_Fraud_Detection_using_PySpark\creditcard.csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True) #header=True to give column names, inferSchema=True to infer the data types of the columns.

    # df.printSchema()

    # df.groupBy('Class').count().orderBy('count').show() #For each class, we see the count of transactions

    fr_df = df.filter(col("Class") == 1) #Class 1 is fraudulent transactions
    nofr_df = df.filter(col("Class") == 0) #Class 0 is non-fraudulent transactions
    ratio = int(nofr_df.count()/fr_df.count())
    # print("ratio: {}".format(ratio))

    oversampled_df = fr_df.withColumn("dummy", explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
    oversampled_df.show()
    # print(oversampled_df.show())
    # print(oversampled_df.count())
    # print(fr_df.count()) #Count is 492, and ratio = 577, so we do 492*577 = 283,524 to get the count of oversampled.


    # Combine both oversampled minority rows and previous majority rows
    df_o = nofr_df.union(oversampled_df)

    cols = df.columns
    cols.remove('Time')
    cols.remove('Class')

    # We specify the object from the VectorAssembler class.
    assembler = VectorAssembler(inputCols=cols, outputCol='features')

    # Now we transform the data into vectors
    data_o = assembler.transform(df_o)

    # data_o.select('features', 'Class').show(truncate=False)
    data_o = data_o.select('features', 'Class')

    minmax_scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features') #Columns of "features" should be scaled and outputted to "scaled_features"
    data_o = minmax_scaler.fit(data_o).transform(data_o) #Computes min and max to scale features, then stores them in scaled_features
    
    
    train_o, test_o = data_o.randomSplit([0.7,0.3]) #Split the data again with the scaled_features

    gradient_boost_class = GBTClassifier(labelCol='Class', featuresCol='scaled_features')
    model_o = gradient_boost_class.fit(train_o)

    predicted_test_gbc_o = model_o.transform(test_o)

    evaluator_gbc = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='accuracy')
    accuracy_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)
    evaluator_gbc.setMetricName("weightedPrecision")
    weightedPrecision_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)
    evaluator_gbc.setMetricName("weightedRecall")
    weightedRecall_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)

    print(f'Gradient Boosted Classifier - Oversampled Data:\n'
        f'  Accuracy:          {accuracy_gbc_o:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_gbc_o:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_gbc_o:.4f}\n')
    # Model path within current directory of project
    # modelPath = "model"
    # model_o.write().overwrite().save(modelPath)
    
    # Convert DenseVector to list using UDF
    # def dense_to_list(dense_vec):
    #     return dense_vec.toArray().tolist()

    # # Register the UDF
    # dense_to_list_udf = udf(dense_to_list, ArrayType(FloatType()))

    # # Apply the UDF to convert DenseVector to list
    # data_o = data_o.withColumn('scaled_features', dense_to_list_udf(data_o['scaled_features']))

    # modelData = json.dumps(data_o.toPandas().to_dict(orient='records'))
    # Make a test prediction
    # newItem = test_o.limit(1)
    # newPrediction = model_o.transform(newItem)
    # newPrediction.select('Class', 'scaled_features', 'prediction').show(truncate=False)

    df = predicted_test_gbc_o.toPandas()

    predictions = list(df['prediction'].values)
    features = list([list(i) for i in df['features'].values])
    modelData = (features, predictions, cols)

