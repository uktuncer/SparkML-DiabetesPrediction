import org.apache.spark.ml.classification.MultiClassSummarizer;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DiabetesModel {
    public static void main(String[] args) {
        SparkSession sparkSession= SparkSession.builder().appName("diabetes-mllib").master("local").getOrCreate();

        Dataset<Row> rawData = sparkSession.read().format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("C:\\Users\\fulli\\Desktop\\diabetes.csv");

        String[] headerList={"Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI"
                ,"DiabetesPedigreeFunction","Age","Outcome"};

        List<String> headers = Arrays.asList(headerList);
        List<String> headersResult=new ArrayList<String>();
        for (String h:headers){
            if (h.equals("Outcome")) {
                StringIndexer indexerTmp=new StringIndexer().setInputCol(h).setOutputCol("label");
                rawData=indexerTmp.fit(rawData).transform(rawData);
                headersResult.add("label");
            }else {
                StringIndexer indexerTmp=new StringIndexer().setInputCol(h).setOutputCol(h.toLowerCase()+"_cat");
                rawData=indexerTmp.fit(rawData).transform(rawData);
                headersResult.add(h.toLowerCase()+"_cat");
            }
        }
        String[] colList = headersResult.toArray(new String[headersResult.size()]);
        VectorAssembler vectorAssembler=new VectorAssembler().setInputCols(colList).setOutputCol("features");
        Dataset<Row> transformData = vectorAssembler.transform(rawData);
        Dataset<Row> finalData = transformData.select("label", "features");

        Dataset<Row>[] datasets = finalData.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData= datasets[1];

        NaiveBayes nb=new NaiveBayes();
        nb.setSmoothing(1);
        NaiveBayesModel model=nb.fit(trainData); // Creating Model

        Dataset<Row> predictions = model.transform(testData);

        MulticlassClassificationEvaluator evaluator=new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double evaluate = evaluator.evaluate(predictions);
    }
}  
