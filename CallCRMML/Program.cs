using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

public class CallPrediction
{
   
    public class CallData
    {
        [LoadColumn(0)]
        public float BranchID { get; set; }

        [LoadColumn(1)]
        public float ClueID { get; set; }

        [LoadColumn(2)]
        public float PersonnelID { get; set; } 
        [LoadColumn(3)]
        public bool Status { get; set; } 

        [LoadColumn(4)]
        public float IsAccount { get; set; }

        [LoadColumn(5)]
        public float HourOfDay { get; set; }

        [LoadColumn(6)]
        public float CreatedYear { get; set; }

        [LoadColumn(7)]
        public float CreatedMonth { get; set; }

        [LoadColumn(8)]
        public float CreatedDay { get; set; }
    }

    public class CallPredictionOutput
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
    }


    public static void Main()
    {
        MLContext mlContext = new MLContext();

        string dataPath = "C:\\Users\\ARKA\\Desktop\\CallCRMML\\Preprocessed_Activities_Time.csv";
        IDataView rawData = mlContext.Data.LoadFromTextFile<CallData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ','
        );

     
        var dataProcessPipeline = mlContext.Transforms.Concatenate(
            "Features",
            "BranchID", "ClueID", "PersonnelID", "IsAccount", "HourOfDay", "CreatedYear", "CreatedMonth", "CreatedDay");

       
        var trainTestData = mlContext.Data.TrainTestSplit(rawData, testFraction: 0.2);
        var trainData = trainTestData.TrainSet;
        var testData = trainTestData.TestSet;

        var trainer = mlContext.BinaryClassification.Trainers.FastTree(
              labelColumnName: "Status", 
              featureColumnName: "Features",
              numberOfLeaves: 20, 
              minimumExampleCountPerLeaf: 10, 
              learningRate: 0.2 
              
          );

     
        var trainingPipeline = dataProcessPipeline.Append(trainer);

     
        Console.WriteLine("Training the model...");
        var model = trainingPipeline.Fit(trainData);

        var predictions = model.Transform(testData);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Status");

        Console.WriteLine($"Accuracy: {metrics.Accuracy}");
        Console.WriteLine($"Precision: {metrics.PositivePrecision}");
        Console.WriteLine($"Recall: {metrics.PositiveRecall}");

        var predictionEngine = mlContext.Model.CreatePredictionEngine<CallData, CallPredictionOutput>(model);

        while (true) 
        {
            Console.WriteLine("Enter BranchID:");
            float branchID = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter ClueID:");
            float clueID = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter PersonnelID:");
            float personnelID = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter IsAccount (1 for Yes, 0 for No):");
            float isAccount = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter HourOfDay:");
            float hourOfDay = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter CreatedYear:");
            float createdYear = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter CreatedMonth:");
            float createdMonth = float.Parse(Console.ReadLine());

            Console.WriteLine("Enter CreatedDay:");
            float createdDay = float.Parse(Console.ReadLine());

            var newCall = new CallData
            {
                BranchID = branchID,
                ClueID = clueID,
                PersonnelID = personnelID,
                IsAccount = isAccount,
                HourOfDay = hourOfDay,
                CreatedYear = createdYear,
                CreatedMonth = createdMonth,
                CreatedDay = createdDay
            };

            var prediction = predictionEngine.Predict(newCall);

            Console.WriteLine("\nPrediction Results:");
            Console.WriteLine($"Predicted Label (Success?): {prediction.PredictedLabel}");
            Console.WriteLine($"Score: {prediction.Score}");

            Console.WriteLine("\nDo you want to enter another data? (y to continue, n to exit):");
            string response = Console.ReadLine().Trim().ToLower();

            if (response != "y")
            {
                Console.WriteLine("Exiting the application. Goodbye!");
                break;
            }
        }

    }
}