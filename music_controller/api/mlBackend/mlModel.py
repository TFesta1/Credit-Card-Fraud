import json
modelData = None

def trainModel(): 
    global modelData
    from pyspark.sql import SparkSession

    # Create a Spark session
    # spark = SparkSession.builder.appName("CreditCardFraudDetection").config('spark.ui.port', "4041").getOrCreate()
    spark = SparkSession.builder.appName("CreditCardFraudDetection").getOrCreate()

    # Test the Spark session
    spark.range(5).show() #Dataframe, column id 0 to 4.

    file_path = r"C:\Users\ringk\OneDrive\Documents\CreditCard_Fraud_Detection_using_PySpark\creditcard.csv"
    df = spark.read.csv(file_path, header=True, inferSchema=True) #header=True to give column names, inferSchema=True to infer the data types of the columns.

    from pyspark.sql.functions import col, count, isnan, when, explode, array, lit
    # col is for df columns
    # when is a conditional expression

    # Counting both null and NaN values for each column
    null_counts = [
        count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
        for c in df.columns
    ]
    """
    Output: We can see the alias being AS c (column name)
    [Column<'count(CASE WHEN ((Time IS NULL) OR isnan(Time)) THEN Time END) AS Time'>,
    Column<'count(CASE WHEN ((V1 IS NULL) OR isnan(V1)) THEN V1 END) AS V1'>,
    ...
    ]
    """


    # Display the count of null and NaN values for each column
    df.select(null_counts).show() #We can see there's 0 null counts
    # Optional: Calculate the total count of null and NaN values across all columns
    total_nulls = sum(
        when(col(c).isNull() | isnan(col(c)), 1).otherwise(0)
        for c in df.columns
    )
    # df.select(total_nulls.alias("Total_Null_or_NaN_Values")).show()

    fr_df = df.filter(col("Class") == 1) #Class 1 is fraudulent transactions
    nofr_df = df.filter(col("Class") == 0) #Class 0 is non-fraudulent transactions
    ratio = int(nofr_df.count()/fr_df.count())
    print("ratio: {}".format(ratio))

    oversampled_df = fr_df.withColumn("dummy", explode(array([lit(x) for x in range(ratio)]))).drop('dummy')
    df_o = nofr_df.union(oversampled_df)

    # Sample the majority class
    sampled_majority_df = nofr_df.sample(False, 1/ratio) #Sample 1/ratio of the majority class, False means random sampling

    # Combine the sampled majority class with the minority class
    df_u = sampled_majority_df.union(fr_df)
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import StandardScaler
    from pyspark.ml.feature import StringIndexer

    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.classification import NaiveBayes
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.classification import RandomForestClassifier

    from pyspark.mllib.evaluation import BinaryClassificationMetrics
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    cols = df.columns
    cols.remove('Time')
    cols.remove('Class')

    # We specify the object from the VectorAssembler class.
    assembler = VectorAssembler(inputCols=cols, outputCol='features')

    # Now we transform the data into vectors
    data_o = assembler.transform(df_o)

    data_o = data_o.select('features', 'Class')
    train_o, test_o = data_o.randomSplit([0.7,0.3])

    # We will do the same prep for undersampled dataframe. But in one box with no displays.

    # df_o is the majority class (oversampling of the non-fraudulent transactions, with the undersampled fraudulent transactions)
    # df_u is the minority class (undersampling of non-fraudulent transactions, with the fraudulent transactions, to make them even)

    # Transform the data into vectors
    data_u = assembler.transform(df_u)

    data_u = data_u.select('features', 'Class')
    train_u, test_u = data_u.randomSplit([0.7,0.3])

    from pyspark.ml.feature import MinMaxScaler
    minmax_scaler = MinMaxScaler(inputCol='features', outputCol='scaled_features') #Columns of "features" should be scaled and outputted to "scaled_features"
    data_o = minmax_scaler.fit(data_o).transform(data_o) #Computes min and max to scale features, then stores them in scaled_features
    data_u = minmax_scaler.fit(data_u).transform(data_u)

    train_o, test_o = data_o.randomSplit([0.7,0.3]) #Split the data again with the scaled_features
    train_u, test_u = data_u.randomSplit([0.7,0.3])

    logReg = LogisticRegression(labelCol='Class', featuresCol='scaled_features', maxIter=40)
    model_o = logReg.fit(train_o)
    model_u = logReg.fit(train_u)
    predicted_test_o = model_o.transform(test_o)
    predicted_test_u = model_u.transform(test_u)

    evaluator = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='accuracy')
    accuracy_LR_o = evaluator.evaluate(predicted_test_o)
    accuracy_LR_u = evaluator.evaluate(predicted_test_u)
    evaluator.setMetricName("weightedPrecision")
    weightedPrecision_LR_o = evaluator.evaluate(predicted_test_o)
    weightedPrecision_LR_u = evaluator.evaluate(predicted_test_u)
    evaluator.setMetricName("weightedRecall")
    weightedRecall_LR_o = evaluator.evaluate(predicted_test_o)
    weightedRecall_LR_u = evaluator.evaluate(predicted_test_u)
    print(f'Logistic Regression - Oversampled Data:\n'
        f'  Accuracy:          {accuracy_LR_o:.4f}\n'
        f'  Weighted Precision:{weightedPrecision_LR_o:.4f}\n'
        f'  Weighted Recall:   {weightedRecall_LR_o:.4f}\n')

    print(f'Logistic Regression - Undersampled Data:\n'
        f'  Accuracy:          {accuracy_LR_u:.4f}\n'
        f'  Weighted Precision:{weightedPrecision_LR_u:.4f}\n'
        f'  Weighted Recall:   {weightedRecall_LR_u:.4f}\n')
    """
    Accuracy: Ratio of correctly predicted outcomes to the total. (TP+TN)/(TP+TN+FP+FN) Where TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
    Weighted Precision: Precision in itself is the ratio of correctly predicted positive observations to the total predicted positive observations. (TP)/(TP+FP). Weighted precision gives more weight to classes with more instances
    Weighted Recall: Recall is the ratio of correctly predicted positives to all observations in the class (TP)/(TP+FN). Weighted recall gives more weight to classes with more instances 


    For each metric, the oversampled model is better because of it's greater values in:
    Accuracy: Generally more correct in predictions
    Weighted Precision: Fewer false positives predictions
    Weighted Recall: Better at capturing true positive instances

    Note The difference between precision and recall
    High precision, low recall: Very selective at predicting positives, but it's correct when it does. Ex: If they say you have a disease, you probably do. It may miss some people with the disease though.
    Low precision, high recall: Tries to capture most of the positives, but it's not very selective/may be wrong. Ex: Captures almost everyone with a disease, but may incorrectly flag some people.

    This is why we need an F1 Score to balance it: F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    Though in some cases, you may prioritize precision, whilst in other cases, recall.

    """


    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Assuming you have Logistic Regression model predictions: predicted_test_o and predicted_test_u

    # Initialize the evaluator for the F1 metric
    evaluator_LR_f1 = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='f1')

    # Evaluate F1 Score for the oversampled dataset
    f1_score_LR_o = evaluator_LR_f1.evaluate(predicted_test_o)

    # Evaluate F1 Score for the undersampled dataset
    f1_score_LR_u = evaluator_LR_f1.evaluate(predicted_test_u)

    # Print results
    print('Logistic Regression - F1 Score Oversampled:', f1_score_LR_o)
    print('Logistic Regression - F1 Score Undersampled:', f1_score_LR_u)


    random_forest_classifier = RandomForestClassifier(labelCol='Class', featuresCol='scaled_features', numTrees=40)
    model_o = random_forest_classifier.fit(train_o)
    model_u = random_forest_classifier.fit(train_u)


    predicted_test_rf_o = model_o.transform(test_o)
    predicted_test_rf_u = model_u.transform(test_u)

    predicted_test_rf_o.select('Class', 'prediction').show(10)
    predicted_test_rf_u.select('Class', 'prediction').show(10)


    evaluator_rf = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='accuracy')
    accuracy_rf_o = evaluator_rf.evaluate(predicted_test_rf_o)
    accuracy_rf_u = evaluator_rf.evaluate(predicted_test_rf_u)
    evaluator_rf.setMetricName("weightedPrecision")
    weightedPrecision_rf_o = evaluator_rf.evaluate(predicted_test_rf_o)
    weightedPrecision_rf_u = evaluator_rf.evaluate(predicted_test_rf_u)
    evaluator_rf.setMetricName("weightedRecall")
    weightedRecall_rf_o = evaluator_rf.evaluate(predicted_test_rf_o)
    weightedRecall_rf_u = evaluator_rf.evaluate(predicted_test_rf_u)
    print(f'Random Forest - Oversampled Data:\n'
        f'  Accuracy:          {accuracy_rf_o:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_rf_o:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_rf_o:.4f}\n')

    print(f'Random Forest - Undersampled Data:\n'
        f'  Accuracy:          {accuracy_rf_u:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_rf_u:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_rf_u:.4f}\n')



    from pyspark.ml.evaluation import MulticlassClassificationEvaluator


    # Initialize the evaluator for the F1 metric
    evaluator_rf_f1 = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='f1')

    # Evaluate F1 Score for the oversampled dataset
    f1_score_rf_o = evaluator_rf_f1.evaluate(predicted_test_rf_o)

    # Evaluate F1 Score for the undersampled dataset
    f1_score_rf_u = evaluator_rf_f1.evaluate(predicted_test_rf_u)

    # Print results
    print('Random Forest - F1 Score Oversampled:', f1_score_rf_o)
    print('Random Forest - F1 Score Undersampled:', f1_score_rf_u)


    naive_bayes = NaiveBayes(featuresCol='scaled_features', labelCol='Class', smoothing=1.0)

    model_o = naive_bayes.fit(train_o)
    model_u = naive_bayes.fit(train_u)
    predicted_test_nb_o = model_o.transform(test_o)
    predicted_test_nb_u = model_u.transform(test_u)

    evaluator_nb = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='accuracy')
    accuracy_NB_o = evaluator_nb.evaluate(predicted_test_nb_o)
    accuracy_NB_u = evaluator_nb.evaluate(predicted_test_nb_u)
    evaluator_nb.setMetricName("weightedPrecision")
    weightedPrecision_NB_o = evaluator_nb.evaluate(predicted_test_nb_o)
    weightedPrecision_NB_u = evaluator_nb.evaluate(predicted_test_nb_u)

    evaluator_nb.setMetricName("weightedRecall")
    weightedRecall_NB_o = evaluator_nb.evaluate(predicted_test_nb_o)
    weightedRecall_NB_u = evaluator_nb.evaluate(predicted_test_nb_u)

    print(f'Naive Bayes - Oversampled Data:\n'
        f'  Accuracy:          {accuracy_NB_o:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_NB_o:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_NB_o:.4f}\n')

    print(f'Naive Bayes - Undersampled Data:\n'
        f'  Accuracy:          {accuracy_NB_u:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_NB_u:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_NB_u:.4f}\n')



    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    # Existing code for fitting the model and predictions
    naive_bayes = NaiveBayes(featuresCol='scaled_features', labelCol='Class', smoothing=1.0)
    model_o = naive_bayes.fit(train_o)
    model_u = naive_bayes.fit(train_u)
    predicted_test_nb_o = model_o.transform(test_o)
    predicted_test_nb_u = model_u.transform(test_u)

    # Use MulticlassClassificationEvaluator with 'f1' metric
    evaluator_nb_f1 = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='f1')

    # Evaluate F1 score
    f1_score_NB_o = evaluator_nb_f1.evaluate(predicted_test_nb_o)
    f1_score_NB_u = evaluator_nb_f1.evaluate(predicted_test_nb_u)

    # Print results
    print('F1 Score Oversampled =', f1_score_NB_o)
    print('F1 Score Undersampled =', f1_score_NB_u)

    gradient_boost_class = GBTClassifier(labelCol='Class', featuresCol='scaled_features')
    model_o = gradient_boost_class.fit(train_o)
    model_u = gradient_boost_class.fit(train_u)

    predicted_test_gbc_o = model_o.transform(test_o)
    predicted_test_gbc_u = model_u.transform(test_u)


    evaluator_gbc = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='accuracy')
    accuracy_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)
    accuracy_gbc_u = evaluator_gbc.evaluate(predicted_test_gbc_u)
    evaluator_gbc.setMetricName("weightedPrecision")
    weightedPrecision_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)
    weightedPrecision_gbc_u = evaluator_gbc.evaluate(predicted_test_gbc_u)
    evaluator_gbc.setMetricName("weightedRecall")
    weightedRecall_gbc_o = evaluator_gbc.evaluate(predicted_test_gbc_o)
    weightedRecall_gbc_u = evaluator_gbc.evaluate(predicted_test_gbc_u)

    print(f'Gradient Boosted Classifier - Oversampled Data:\n'
        f'  Accuracy:          {accuracy_gbc_o:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_gbc_o:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_gbc_o:.4f}\n')

    print(f'Gradient Boosted Classifier - Undersampled Data:\n'
        f'  Accuracy:          {accuracy_gbc_u:.4f}\n'
        f'  Weighted Precision: {weightedPrecision_gbc_u:.4f}\n'
        f'  Weighted Recall:    {weightedRecall_gbc_u:.4f}\n')

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator



    # Initialize the evaluator for the F1 metric
    evaluator_gbc_f1 = MulticlassClassificationEvaluator(labelCol='Class', predictionCol='prediction', metricName='f1')

    # Evaluate F1 Score for the oversampled dataset
    f1_score_gbc_o = evaluator_gbc_f1.evaluate(predicted_test_gbc_o)

    # Evaluate F1 Score for the undersampled dataset
    f1_score_gbc_u = evaluator_gbc_f1.evaluate(predicted_test_gbc_u)

    # Print results
    print('Gradient Boosted Classifier - F1 Score Oversampled:', f1_score_gbc_o)
    print('Gradient Boosted Classifier - F1 Score Undersampled:', f1_score_gbc_u)

    from tabulate import tabulate

    # Headers for the table
    headers = ["Model", "Data Type", "Accuracy", "Weighted Precision", "Weighted Recall", "F1 Score"]

    # Data for the table
    data = [
        ["Naive Bayes", "Oversampled", accuracy_NB_o, weightedPrecision_NB_o, weightedRecall_NB_o, f1_score_NB_o],
        ["Naive Bayes", "Undersampled", accuracy_NB_u, weightedPrecision_NB_u, weightedRecall_NB_u, f1_score_NB_u],
        ["Gradient Boosted", "Oversampled", accuracy_gbc_o, weightedPrecision_gbc_o, weightedRecall_gbc_o, f1_score_gbc_o],
        ["Gradient Boosted", "Undersampled", accuracy_gbc_u, weightedPrecision_gbc_u, weightedRecall_gbc_u, f1_score_gbc_u],
        ["Random Forest", "Oversampled", accuracy_rf_o, weightedPrecision_rf_o, weightedRecall_rf_o, f1_score_rf_o],
        ["Random Forest", "Undersampled", accuracy_rf_u, weightedPrecision_rf_u, weightedRecall_rf_u, f1_score_rf_u],
        ["Logistic Regression", "Oversampled", accuracy_LR_o, weightedPrecision_LR_o, weightedRecall_LR_o, f1_score_LR_o],
        ["Logistic Regression", "Undersampled", accuracy_LR_u, weightedPrecision_LR_u, weightedRecall_LR_u, f1_score_LR_u]
    ]
    tabulatedData = tabulate(data, headers=headers)
    # Print the table
    print(tabulatedData)

    def calculateScores(accuracy, weighted_precision, weighted_recall, f1_score):
        return (accuracy + weighted_precision + weighted_recall + f1_score) / 4

    def getBestModel(model_name, data_type, accuracy, weighted_precision, weighted_recall, f1_score, models):
        """
        This function returns the best model based on the F1 Score
        """
        # Initialize the best model
        best_model = None

        # Initialize the best F1 Score
        best_score = 0

        # Iterate through the data
        for i in range(len(model_name)):
            score = calculateScores(accuracy[i], weighted_precision[i], weighted_recall[i], f1_score[i])
            # Check if the F1 Score is greater than the best F1 Score
            if score > best_score:
                # Update the best F1 Score
                best_score = score

                # Update the best model
                best_model = (model_name[i], data_type[i], accuracy[i], weighted_precision[i], weighted_recall[i], f1_score[i], models[i])

        # Return the best model
        return best_model

    # Get the best model
    best_model = getBestModel(
        model_name=["Naive Bayes", "Naive Bayes", "Gradient Boosted", "Gradient Boosted", "Random Forest", "Random Forest", "Logistic Regression", "Logistic Regression"],
        data_type=["Oversampled", "Undersampled", "Oversampled", "Undersampled", "Oversampled", "Undersampled", "Oversampled", "Undersampled"],
        accuracy=[accuracy_NB_o, accuracy_NB_u, accuracy_gbc_o, accuracy_gbc_u, accuracy_rf_o, accuracy_rf_u, accuracy_LR_o, accuracy_LR_u],
        weighted_precision=[weightedPrecision_NB_o, weightedPrecision_NB_u, weightedPrecision_gbc_o, weightedPrecision_gbc_u, weightedPrecision_rf_o, weightedPrecision_rf_u, weightedPrecision_LR_o, weightedPrecision_LR_u],
        weighted_recall=[weightedRecall_NB_o, weightedRecall_NB_u, weightedRecall_gbc_o, weightedRecall_gbc_u, weightedRecall_rf_o, weightedRecall_rf_u, weightedRecall_LR_o, weightedRecall_LR_u],
        f1_score=[f1_score_NB_o, f1_score_NB_u, f1_score_gbc_o, f1_score_gbc_u, f1_score_rf_o, f1_score_rf_u, f1_score_LR_o, f1_score_LR_u],
        models=[predicted_test_nb_o, predicted_test_nb_u, predicted_test_gbc_o, predicted_test_gbc_u, predicted_test_rf_o, predicted_test_rf_u, predicted_test_o, predicted_test_u]
    )



    # Example data
    models = ['Naive Bayes', 'Gradient Boosted', 'Random Forest', 'Logistic Regression']
    accuracy_oversampled = [accuracy_NB_o, accuracy_gbc_o, accuracy_rf_o, accuracy_LR_o]
    precision_oversampled = [weightedPrecision_NB_o, weightedPrecision_gbc_o, weightedPrecision_rf_o, weightedPrecision_LR_o]
    recall_oversampled = [weightedRecall_NB_o, weightedRecall_gbc_o, weightedRecall_rf_o, weightedRecall_LR_o]

    import matplotlib.pyplot as plt
    import numpy as np

    def makeModelPerformance(models, accuracy, precision, recall, undersampled=False):
        # Setting the positions and width for the bars
        pos = list(range(len(models)))
        width = 0.25

        # Plotting the bars
        fig, ax = plt.subplots(figsize=(20, 10)) #10,5

        # Create a bar with accuracy data,
        # in position pos,
        plt.bar(pos,
                accuracy,
                width,
                alpha=0.5,
                color='#EE3224',
                label="Accuracy")

        # Create a bar with precision data,
        # in position pos + some width buffer,
        plt.bar([p + width for p in pos],
                precision,
                width,
                alpha=0.5,
                color='#F78F1E',
                label="Precision")

        # Create a bar with recall data,
        # in position pos + width buffer,
        plt.bar([p + width*2 for p in pos],
                recall,
                width,
                alpha=0.5,
                color='#FFC222',
                label="Recall")

        # Set the y axis label
        ax.set_ylabel('Score')

        # Set the chart's title
        ax.set_title(f"Model Performance ({'Undersampled' if undersampled else 'Oversampled'} Data)", fontsize=30)

        # Set the position of the x ticks
        ax.set_xticks([p + 1.5 * width for p in pos])

        # Set the labels for the x ticks
        ax.set_xticklabels(models, fontsize=20)

        # Adding the legend and showing the plot
        plt.legend(['Accuracy', 'Precision', 'Recall'], loc='upper left', fontsize=15)
        plt.grid()

    # import matplotlib.pyplot as plt

    # # Your plotting code here
    # plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    makeModelPerformance(models, accuracy_oversampled, precision_oversampled, recall_oversampled)
    # Instead of plt.show(), use plt.savefig()
    plt.savefig(r'C:\Users\ringk\OneDrive\Documents\Credit-Card-Fraud\music_controller\frontend\static\images\ModelPerformance (Oversampled).png')

    accuracy_undersampled = [accuracy_NB_u, accuracy_gbc_u, accuracy_rf_u, accuracy_LR_u]
    precision_undersampled = [weightedPrecision_NB_u, weightedPrecision_gbc_u, weightedPrecision_rf_u, weightedPrecision_LR_u]
    recall_undersampled = [weightedRecall_NB_u, weightedRecall_gbc_u, weightedRecall_rf_u, weightedRecall_LR_u]

    makeModelPerformance(models, accuracy_undersampled, precision_undersampled, recall_undersampled, undersampled=True)

    # Instead of plt.show(), use plt.savefig()
    plt.savefig(r'C:\Users\ringk\OneDrive\Documents\Credit-Card-Fraud\music_controller\frontend\static\images\ModelPerformance (Undersampled).png')

    # This will save the plot as a PNG file named 'your_plot.png' in the current working directory

    # Example model names
    models = ['Naive Bayes', 'Gradient Boosted', 'Random Forest', 'Logistic Regression']

    # Example F1 scores for each model (replace with your actual scores)
    f1_scores_oversampled = [f1_score_NB_o, f1_score_gbc_o, f1_score_rf_o, f1_score_LR_o] # Replace with your F1 scores for oversampled
    f1_scores_undersampled = [f1_score_NB_u, f1_score_gbc_u, f1_score_rf_u, f1_score_LR_u] # Replace with your F1 scores for undersampled

    # Setting the positions for the bars
    pos = list(range(len(models)))
    width = 0.35  # Width of a bar

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    plt.bar([p - width/2 for p in pos],
            f1_scores_oversampled,
            width,
            alpha=0.5,
            color='#EE3224',
            label='Oversampled')

    plt.bar([p + width/2 for p in pos],
            f1_scores_undersampled,
            width,
            alpha=0.5,
            color='#F78F1E',
            label='Undersampled')

    # Setting axis labels, title, and ticks
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score by Model and Sampling Method', fontsize=30)
    ax.set_xticks(pos)
    ax.set_xticklabels(models)

    # Adding the legend and showing the plot
    plt.legend(['Oversampled', 'Undersampled'], loc='upper right')
    plt.grid()

    plt.savefig(r'C:\Users\ringk\OneDrive\Documents\Credit-Card-Fraud\music_controller\frontend\static\images\F1Score.png')






    # Drop all rows of df but the last 10
    chosenModel = best_model[-1].toPandas()
    best_model_details = best_model[:-1]

    chosenModel = chosenModel.sample(frac=1).reset_index(drop=True)
    chosenModel = chosenModel.tail(10)





    predictions = list(chosenModel['prediction'].values)
    features = list([list(i) for i in chosenModel['features'].values])

    total_transactions = df.count()

    # Total number of columns
    total_columns = len(df.columns)

    # Total number of features (assuming the last column is the label)
    total_features = total_columns - 1

    # Total number of label(s)
    total_labels = 1

    # Total number of normal transactions
    total_normal_transactions = df.filter(col('Class') == 0).count()

    # Total number of fraudulent transactions
    total_fraudulent_transactions = df.filter(col('Class') == 1).count()

    # Percentage of fraudulent transactions
    percentage_fraudulent = (total_fraudulent_transactions / total_transactions) * 100

    # Percentage of normal transactions
    percentage_normal = (total_normal_transactions / total_transactions) * 100

    print("Total number of transactions:", total_transactions)
    print("Total number of columns:", total_columns)
    print("Total number of features:", total_features)
    print("Total number of label(s):", total_labels)
    print("Total number of normal transactions:", total_normal_transactions)
    print("Total number of fraudulent transactions:", total_fraudulent_transactions)
    print("Percentage of fraudulent transactions: {:.4f}%".format(percentage_fraudulent))
    print("Percentage of normal transactions: {:.4f}%".format(percentage_normal))

    # We'll return all of this info on the front-end to display
    datasetInfo = [total_transactions, total_columns, total_features, total_labels, total_normal_transactions, total_fraudulent_transactions, "{:.4f}%".format(percentage_fraudulent), "{:.4f}%".format(percentage_normal)]
        

    modelData = (features, predictions, cols, datasetInfo, best_model_details, tabulatedData)

