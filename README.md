#AI_FINAL_PROJECT
<h2>Hi I’m Stroke Prediction Model</h2>

I focus on predicting whether a patient is likely to have a stroke based on various input features such as gender, age, presence of diseases, and smoking status. The goal is to build a predictive model that can assist in identifying individuals who may be at a higher risk of experiencing a stroke. 


<h3>Project Phases</h3>

<h3>Phase 1: Data Analysis & Preparation</h3>

In this phase, the project performs an initial analysis of the dataset and prepares the data for further processing. Key steps include:

Importing necessary Python libraries for data analysis and visualization.
Loading the dataset from a CSV file.
Gaining insights into the dataset, such as checking for missing values, understanding data statistics, and exploring the distribution of the target variable (stroke).
Performing data cleaning by removing rows with missing values and visualizing the distribution of features using density plots and count plots.

<h3>Phase 2: Building an Overfitting Model</h3>

In this phase, the project builds a neural network model that overfits the entire dataset. This step aims to understand the model's behavior and evaluate its performance. The steps involved are:

<li>Building a sequential neural network model with multiple layers and appropriate activation functions.</li>
<li>Compiling the model with the binary cross-entropy loss function and the Adam optimizer.</li>
<li>Training the model on the entire dataset and monitoring its accuracy over multiple epochs.</li>
<li>Analyzing the model's accuracy using line plots to identify potential overfitting or convergence.</li>

<h3>Phase 3: Model Selection & Evaluation</h3>

In this phase, the project focuses on selecting the best model architecture and evaluating its performance using training and validation datasets. The steps involved are:

<li>Shuffling the dataset to ensure random distribution of data.</li>
<li>Splitting the dataset into training and validation sets.</li>
<li>Building multiple neural network models with different architectures and activation functions.</li>
<li>Compiling the models with appropriate loss functions and optimizers.</li>
<li>Training the models on the training dataset while monitoring their accuracy on the validation dataset.</li>
<li>Evaluating the models' accuracy and selecting the best-performing model based on the validation accuracy.</li>
<li>Visualizing the accuracy of each model over multiple epochs using line plots.</li>


<h3>Usage</h3>

To use this project, follow these steps:

<li>Install the necessary libraries mentioned in the requirements.txt file.</li>
<li>Download the dataset (stroke_data.csv) and place it in the same directory as the project files.</li>
<li>Run the Python script or notebook file to execute the code.</li>
<li>Follow the instructions and comments within the code to understand each phase and its output.</li>
<li>Analyze the results, including data insights, model performance, and accuracy trends.</li>
<li>Modify the code or experiment with different model architectures to further improve the accuracy.</li>

<h3>Dependencies</h3>

This project requires the following libraries:

numpy
pandas
seaborn
matplotlib
missingno
keras (TensorFlow)

You can install the required libraries using pip:

<h4>pip install “library name”</h4>

Note: It is recommended to use a virtual environment to manage the project dependencies.



<h3>Dataset</h3>

The dataset used in this project is stored in the "stroke_data.csv" file. It contains important details about individuals, including age, gender, smoking status, and stroke occurrence. The dataset should be placed in the same directory as the project files.

<h3>Conclusion</h3>

The README file provides an overview of the project, this project developed a predictive model for stroke prediction based on various input features. The project involved data analysis, model development, and evaluation. To use the project, install the required libraries and download the dataset (stroke_data.csv). Execute the provided Python script or notebook to analyze the results and modify the code as needed. The project's goal is to assist healthcare professionals and researchers in identifying individuals at a higher risk of stroke. By leveraging this model, preventive measures and personalized treatment plans can be implemented.

<h4>Thank you for your interest in this project, and feel free to reach out with any questions or feedback.</h4>
