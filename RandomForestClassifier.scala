import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import org.apache.spark.mllib.evaluation.RegressionMetrics

//file is export of customer data, use spark ML to figure out if customers will be CHRONIC_BAD_PAY customers
val nameRDD = sc.textFile("/home/hadoop/temp/filename.csv")

case class Individual(INDIV_ID: String, CHRONIC_BAD_PAY: Double, BUSINESS_FLG: Double, DM_PROM_HIST_FLG: Double, 
REN_PROM_HIST_FLG: Double, BILL_PROM_HIST_FLG: Double, BEST_GENDER: Double)


def parseIndiv(str: String): Individual = {
      val p = str.split(",")
      Individual(p(0), p(1).toDouble,  p(2).toDouble, p(3).toDouble, p(4).toDouble, p(5).toDouble,p(6).toDouble)}


val name2RDD = nameRDD.map(parseIndiv)
name2RDD.first
name2RDD.take(5)
name2RDD.count()

val nameDF =  nameRDD.map(parseIndiv).toDF().cache()
val totalunique = nameDF.select("INDIV_ID").distinct.count
val featureCols = Array( "BUSINESS_FLG", "DM_PROM_HIST_FLG", "REN_PROM_HIST_FLG", "BILL_PROM_HIST_FLG", "BEST_GENDER")

val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

val df2 = assembler.transform(nameDF)

df2.show(2)

val labelIndexer = new StringIndexer().setInputCol("CHRONIC_BAD_PAY").setOutputCol("label")
val df3 = labelIndexer.fit(df2).transform(df2)
df3.show(2)

val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3),5043)
val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(5043)

//impurity:Criterion used for information gain calculation
//maxDepth: Maximum depth of a tree. Increasing the depth makes the model more powerful, but deep trees take longer to train.
//Increasing the depth makes the model more expressive and powerful.
//maxBins: Maximum number of bins used for discretizing continuous features and for choosing how to split on features at each node.
//Increasing the number of trees will decrease the variance in predictions, improving the model’s test-time accuracy.
//auto:Automatically select the number of features to consider for splits at each tree node
//seed:Use a random seed number , allowing to repeat the results

//train model
val model = classifier.fit(trainingData)  

// make predictions
val predictions = model.transform(testData)

// model transform produced  new columns: rawPrediction, probablity and prediction.

predictions.show(1)
predictions.count
predictions.printSchema()

val pdata =predictions.select(predictions.col("INDIV_ID"), predictions.col("CHRONIC_BAD_PAY"),predictions.col("BUSINESS_FLG"),
predictions.col("DM_PROM_HIST_FLG"),predictions.col("REN_PROM_HIST_FLG"),predictions.col("BILL_PROM_HIST_FLG"),predictions.col("BEST_GENDER"), predictions.col("label"), predictions.col("prediction"))

pdata.write.format("csv").option("header", "false").save("/home/hadoop/temp/1")
predictions.write.option("compression","none").mode("overwrite").save("/home/hadoop/temp/1")

val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
val accuracy = evaluator.evaluate(predictions)

val predictionAndLabels =predictions.select("prediction", "label").rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

println("area under the precision-recall curve: " + metrics.areaUnderPR)
println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)

// Save and load model
model.save(sc, "target/tmp/myRandomForestClassificationModel")
val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
