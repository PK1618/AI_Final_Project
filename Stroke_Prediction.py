#!/usr/bin/env python
# coding: utf-8

# <b><h3>Introduction</h3></b>

# Based on input features like gender, age, various diseases, and smoking status, this dataset is used to predict whether a patient is likely to get a stroke. 
# 
# About the Data: Each row in the data contains important details about an individual, such as age, gender, smoking status, and stroke occurrence.

# <br></br>
# <b><h2>Phase1: Data analysis & preparation</h2></b>

# <br></br>
# <b><h2>Importing Python Libraries</h2></b>

# In[1]:


import random
import numpy as np
from pprint import pprint
import pandas as pd
import seaborn as sns
import missingno as msno
import os

import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint


# <br></br>
# <b><h2>Loading the Data</h2></b>

# In[2]:


# Load dataset
df = pd.read_csv('stroke_data.csv', delimiter = ',')
print(df.shape)
df.head(10)


# <br></br>
# <b><h2>Insights</h2></b>

# In[3]:


df.info()


# In[4]:


df.describe(include="all")


# In[5]:


# df.loc[:, "DeadlineRespected"].value_counts(normalize=True) * 100
df.loc[:, "stroke"].value_counts()


# In[6]:


(df.loc[:, "stroke"] == 1).sum()


# <br></br>
# <b><h2>Data Cleaning</h2></b>

# In[7]:


# Check for missing data and duplicates
df.isnull().sum()
df.dropna(inplace = True)
df.shape
# print('Duplicates:', df.duplicated().sum())


# In[8]:


msno.bar(df)


# In[9]:


# Select the 11 features to plot
features = ["sex", "age", "hypertension", "heart_disease", "ever_married", 
            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"]

# Set the plot style
sns.set_style("darkgrid")

# Create a subplot for each feature
fig, axs = plt.subplots(11, 1, figsize=(8, 30))

# Loop through each feature and plot the density chart
for i, feature in enumerate(features):
    sns.kdeplot(data=df, x=feature, ax=axs[i], fill=True, alpha=0.5)
    axs[i].set_xlabel(feature, fontsize=14)
    axs[i].set_ylabel("Density", fontsize=14)
    axs[i].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()


# In[10]:


sns.countplot(data = df, x="stroke")


# In[11]:


percentage = df.loc[:, "stroke"].value_counts(normalize=True) * 100
percentage


# <b> Pie chart distribution of DeadlineRespected label </b>

# In[12]:


plt.pie(percentage, labels=["0", "1"], autopct = "%1.1f%%")
plt.title("Distribution of Stroke label")
plt.show()


# In[13]:


#Visualization before Normalization
df.hist(figsize=(12,15))
plt.show()


# Check balance of output labels
above_mean = (df['stroke'] > df['stroke'].mean()).sum()
below_mean = len(df) - above_mean
print('Above mean:', above_mean/len(df))
print('Below mean:', below_mean/len(df))

# # Normalize data
# min_val = min(df_new)
# max_val = max(df_new)
    
# # # Calculate the range of the data
# # data_range = max_val - min_val
    
    # Normalize the data
def z_score_normalization(column_name):
    series = df.loc[:, column_name]
    return (series - series.mean())/series.std()
    
# Iterates over all the continuous columns and applies z_score_normalization to each column
for column_name in df.columns:
    if column_name != "stroke":
        df[column_name] = z_score_normalization(column_name=column_name)

#Visualization after Normalization
df.hist(figsize=(12,15))
plt.show()


# In[14]:


print(df.shape)


# In[15]:


df.describe(include="all")


# In[16]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# <br></br>
# <b><h2>Phase2: Build a model to overfit the entire dataset</h2></b>

# In[148]:


# df_new = df.sample(frac=0.6)


# In[232]:


# X = df_new.iloc[:, :-1].values
# y = df_new.iloc[:, -1].values


# In[20]:


model = Sequential()

model.add(Dense(1, input_dim = df.shape[1] -1, activation = 'sigmoid'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(1, activation = 'sigmoid'))


# In[21]:


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# checkpoint_file
# print(model.summary())


# In[22]:


history = model.fit(X, y, epochs=100, verbose=1)


# In[23]:


history.history["accuracy"][99]


# In[24]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'Accuracy': history.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[123]:


model1 = Sequential()

model1.add(Dense(8, activation="relu", input_dim=df.shape[1] - 1))
model1.add(Dense(4, activation="relu"))
model1.add(Dense(1, activation="sigmoid"))

model1.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[124]:


history1 = model1.fit(X, y, epochs=100, verbose=1)


# In[125]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'accuracy': history1.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data1)

plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[29]:


model2 = Sequential()

model2.add(Dense(32, activation="relu", input_dim=df.shape[1] - 1))
model2.add(Dense(16, activation="relu"))
model2.add(Dense(8, activation="relu"))
model2.add(Dense(1, activation="sigmoid"))

model2.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[30]:


history2 = model2.fit(X, y, epochs=100, verbose=1)


# In[31]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'accuracy': history2.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data1)

plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[126]:


model3 = Sequential()

model3.add(Dense(40, activation="relu", input_dim=df.shape[1] - 1))
model3.add(Dense(18, activation="relu"))
model3.add(Dense(8, activation="relu"))
model3.add(Dense(4, activation="relu"))
model3.add(Dense(1, activation="sigmoid"))


model3.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[127]:


history3 = model3.fit(X, y, epochs=100, verbose=1)


# In[129]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'accuracy': history3.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data1)

plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[130]:


model4 = Sequential()

model4.add(Dense(128, activation="tanh", input_dim=df.shape[1] - 1))
model4.add(Dense(64, activation="tanh"))
model4.add(Dense(32, activation="tanh"))
model4.add(Dense(16, activation="tanh"))
model4.add(Dense(8, activation="tanh"))
model4.add(Dense(1, activation="sigmoid"))

model4.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[131]:


history4 = model4.fit(X, y, epochs=100, verbose=1)


# In[132]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'accuracy': history4.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data1)

plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# In[133]:


modelx = Sequential()

modelx.add(Dense(128, activation="tanh", input_dim=df.shape[1] - 1))
modelx.add(Dense(80, activation="tanh"))
modelx.add(Dense(64, activation="tanh"))
modelx.add(Dense(32, activation="tanh"))
modelx.add(Dense(16, activation="tanh"))
modelx.add(Dense(8, activation="tanh"))
modelx.add(Dense(1, activation="sigmoid"))

modelx.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# In[134]:


historyx = modelx.fit(X, y, epochs=100, verbose=1)


# In[135]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'accuracy': history4.history['accuracy']})

plt.figure(figsize=(16, 10))

sns.lineplot(data=data1)

plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()


# <h5>Observation</h5><br>
# 
# 
# The model4 has given the best accuracy, a neural network with 6 layers for binary classification. It has 128 neurons in the input layer and uses the 'tanh' activation function. The 4 hidden layers have 64, 32, 16 and 8 neurons, respectively, and also use the 'tanh' activation function. The output layer has a single neuron and uses a 'sigmoid' activation function. The model uses binary cross-entropy loss function, Adam optimizer, and accuracy as the metric. The use of the 'tanh' activation function in the hidden layers suggests that the model can capture non-linear relationships between input features and the stroke(Output) variable.
# 
# The model4 shows the signs of better overfit and convergence at epochs 100 which has accuracy of 99.27 and there is no futher improvement in the accuracy(modelx).
# 

# <br></br>
# <b><h2>Phase 3: Model selection & evaluation</h2></b>

# <b><h4>Data Shuffling</h4></b>

# In[39]:


df = df.sample(frac=1).reset_index(drop=True)
df.head(10)


# <b><h4>Data Splitting into TRAINING AND VALIDATION</h4></b>

# In[42]:


index_20percent = int(0.2 * len(df.iloc[:, 0].values))
print(index_20percent)

XVALIDATION = df.iloc[:index_20percent, :-1].values
YVALIDATION = df.iloc[:index_20percent, -1].values

XTRAIN = df.iloc[index_20percent:, 0:-1].values
YTRAIN = df.iloc[index_20percent:, -1].values


# <b><h4>Single Layer Model</h4></b>

# In[43]:


model5 = Sequential()

model5.add(Dense(1, input_dim=XTRAIN.shape[1], activation = "relu"))


# In[45]:


model5.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[46]:


history5 = model5.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[110]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history5.history['accuracy'], 'val_accuracy': history5.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history5.history['loss'], 'val_loss': history5.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[48]:


model6 = Sequential()

model6.add(Dense(8, input_dim=XTRAIN.shape[1], activation = "relu"))
model6.add(Dense(4, activation = "relu"))
model6.add(Dense(1, activation = "sigmoid"))


# In[49]:


XTRAIN.shape, XVALIDATION.shape


# In[50]:


model6.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_model_1.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[51]:


history6 = model6.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[111]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history6.history['accuracy'], 'val_accuracy': history6.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history6.history['loss'], 'val_loss': history6.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[54]:


model7 = Sequential()

model7.add(Dense(32, input_dim=XTRAIN.shape[1], activation = "relu"))
model7.add(Dense(16, activation = "relu"))
model7.add(Dense(8, activation = "relu"))
model7.add(Dense(1, activation = "sigmoid"))


# In[55]:


model7.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[56]:


history7 = model7.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[112]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history7.history['accuracy'], 'val_accuracy': history7.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history7.history['loss'], 'val_loss': history7.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[58]:


model8 = Sequential()

model8.add(Dense(36, input_dim=XTRAIN.shape[1], activation = "relu"))
model8.add(Dense(18, activation = "relu"))
model8.add(Dense(8, activation = "relu"))
model8.add(Dense(1, activation = "sigmoid"))


# In[59]:


model8.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_point.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[60]:


history8 = model8.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[113]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history8.history['accuracy'], 'val_accuracy': history8.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history8.history['loss'], 'val_loss': history8.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[62]:


model9 = Sequential()

model9.add(Dense(64, input_dim=XTRAIN.shape[1], activation = "tanh"))
model9.add(Dense(32, activation = "tanh"))
model9.add(Dense(16, activation = "tanh"))
model9.add(Dense(8, activation = "tanh"))
model9.add(Dense(1, activation = "sigmoid"))


# In[63]:


model9.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_position.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[64]:


history9 = model9.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[65]:


accuracy_allFeatures = history9.history['val_accuracy'][99]
accuracy_allFeatures


# In[114]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history9.history['accuracy'], 'val_accuracy': history9.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history9.history['loss'], 'val_loss': history9.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[67]:


model10 = Sequential()

model10.add(Dense(128, input_dim=XTRAIN.shape[1], activation = "tanh"))
model10.add(Dense(64, activation = "tanh"))
model10.add(Dense(32, activation = "tanh"))
model10.add(Dense(16, activation = "tanh"))
model10.add(Dense(8, activation = "tanh"))
model10.add(Dense(1, activation = "sigmoid"))


# In[68]:


model10.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint("best_Weights_of_model10.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# In[69]:


history10 = model10.fit(XTRAIN, YTRAIN, epochs=100, validation_data=(XVALIDATION, YVALIDATION), callbacks=[checkpoint_callback])


# In[115]:


# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data = pd.DataFrame({'accuracy': history10.history['accuracy'], 'val_accuracy': history10.history['val_accuracy']})

# Create a pandas DataFrame with "accuracy" and "Val accuracy" columns
data1 = pd.DataFrame({'loss': history10.history['loss'], 'val_loss': history10.history['val_loss']})

# Create a figure with two subplots
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the first subplot
sns.lineplot(data=data, ax=axes[0])
axes[0].set_title('Model accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')

# Plot the second subplot
sns.lineplot(data=data1, ax=axes[1])
axes[1].set_title('Model loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')

# Show the plot
plt.show()


# In[140]:


from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# In[157]:


from sklearn.metrics import classification_report

model_names = ["model5", "model6", "model7", "model9", "model10"]
model_weights = ["best_model.hdf5", "best_model_1.hdf5", "best_weights.hdf5", "best_position.hdf5", "best_Weights_of_model10.hdf5"]
for model_name in model_names:
    model = globals()[model_name]
    for model_weight in model_weights:
#         weight = globals()[model_weight]
        # Load the saved best weights
        model.load_weights(model_weight)

        # Make predictions on the validation set
        YVALIDATION_PREDICTED = model.predict(XVALIDATION)
        YVALIDATION_PREDICTED = [1 if y>=0.5 else 0 for y in YVALIDATION_PREDICTED]

        # Generate classification report
        print(classification_report(YVALIDATION, YVALIDATION_PREDICTED))


# In[143]:


models = ["model5", "model6", "model7", "model9", "model10"]

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

for i, model_name in enumerate(models):
    model = globals()[model_name]
    fpr, tpr, thresholds = roc_curve(YVALIDATION, model.predict(XVALIDATION))
    roc_auc = auc(fpr, tpr)
#     precision = precision_score(YVALIDATION, model.predict(XVALIDATION))
#     recall = recall_score(YVALIDATION, model.predict(XVALIDATION))
#     f1 = f1_score(YVALIDATION, model.predict(XVALIDATION))
    
    row = i // 2
    col = i % 2
    axs[row, col].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axs[row, col].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[row, col].set_xlabel('False Positive Rate')
    axs[row, col].set_ylabel('True Positive Rate')
    axs[row, col].set_title(f'{model_name} (ROC) Curve')
    axs[row, col].legend(loc="lower right")
    
#     print(f'Precision: {precision:.2f}')
#     print(f'Recall: {recall:.2f}')
#     print(f'F1 score: {f1:.2f}')

plt.delaxes(axs[2][1])
plt.subplots_adjust(hspace=0.5)

plt.show()


# <h5>Observation</h5><br>
# 
# <li>The above model of a training dataset gives the best Validation accuracy of 99.56 at 100th epoch, which has 6 layers.</li>
# <li>Input layer consists of 128 neurons and "tanh" as an activation function.</li>
# <li>The other 4 Dense layers have 64, 32, 16 and 8 neurons respectively with activation function "tanh".
# Output layer has single neuron with activation function "sigmoid"
# 
#     
# Sticking with the accuracy as there is no further improvement. 

# <h2>Phase 4: Feature importance and reduction</h2>

# In[74]:


model10.load_weights("best_Weights_of_model10.hdf5")


# In[75]:


df.shape


# In[76]:


from sklearn import metrics


def feature_importance():
    # Let's initialize the list to store the feature wise accuracy
    feature_accuracy = {}

    for index in range(XTRAIN.shape[1]):
        # Let's get the values corresponding to the each feature
        single_input_feature = XTRAIN[:, index]
        valid_input_feature = XVALIDATION[:, index]

        single_feature_model = Sequential()
        single_feature_model.add(Dense(128, input_dim=1, activation = "tanh"))
        single_feature_model.add(Dense(64, input_dim=1, activation = "tanh"))
        single_feature_model.add(Dense(32, input_dim=1, activation = "tanh"))
        single_feature_model.add(Dense(16, input_dim=1, activation = "tanh"))
        single_feature_model.add(Dense(8, input_dim=1, activation = "tanh"))
        single_feature_model.add(Dense(1, activation = "sigmoid"))

        # Let's build the model using binary_crossentropy as the loss function 
        # and accuracy as the evaluation metric during the compilation process
        single_feature_model.compile(loss="binary_crossentropy", optimizer = "adam", metrics=["accuracy"])

        callback_a = ModelCheckpoint("model_feature1.hdf5", monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0)
#         callback_b = EarlyStopping(monitor="val_loss", mode="min", patience=20, verbose=0)

        print(f"Let's fit the sequential model on {df.columns[index]}")
        # Let's fit the sequential model with input features and output label
        history = single_feature_model.fit(
            x=single_input_feature, 
            y=YTRAIN, 
            validation_data=(valid_input_feature, YVALIDATION), 
            epochs=100, 
            batch_size=100, 
            callbacks=[callback_a], 
            verbose=0
        )
        # Let's predict on the validation inputs
        hypothesis = single_feature_model.predict(valid_input_feature, verbose=0)
        hypothesis = (hypothesis > 0.5)
        accuracy_score = metrics.accuracy_score(YVALIDATION, hypothesis)
        feature_accuracy[df.columns[index]] = accuracy_score

        print(f"Accuracy score corresponding to {df.columns[index]} --> {accuracy_score}")
        print()
    return feature_accuracy


# In[77]:


# Let's call the feature_importance function to calulate the importance of each feature
feature_accuracy = feature_importance()

# Let's print the feature_accuracy dictionary
feature_accuracy


# In[78]:


# Let's plot the barchart of different features on
# X-axis and corresponding accuracies on y-axis

sorted_feature_accuracy = {key: value for key, value in sorted(feature_accuracy.items(), key=lambda item: item[1])}
print(sorted_feature_accuracy)
sorted_features = np.array(list(sorted_feature_accuracy.keys()))
print(sorted_features)
plt.figure(figsize=(20, 7))

# Let's convert the acuuracies into percentages
feature_acc = np.array(list(sorted_feature_accuracy.values())) * 100

sns.barplot(x=list(sorted_feature_accuracy.keys()), y=feature_acc, palette="hls")
plt.show()


# In[79]:


list(df.columns.values)


# In[86]:


# Load dataset
df = pd.read_csv('stroke_data.csv', delimiter = ',')
print(df.shape)
df.head(10)
train_df = pd.DataFrame(XTRAIN, columns=df.columns.tolist()[:-1])
valid_df = pd.DataFrame(XVALIDATION, columns=df.columns.tolist()[:-1])


# In[87]:


def feature_importance_and_reduction():
    accuracy_after_each_reduction = []
    Val_accuracy_after_each_reduction = []
    for index, feature in enumerate(sorted_features):
        
        if index == len(sorted_features):
            break
        train_df.drop(feature, axis=1, inplace=True)
        valid_df.drop(feature, axis=1, inplace=True)
        model_1 = Sequential()
        
        model_1.add(Dense(128, input_dim=train_df.shape[1], activation = "tanh"))
        model_1.add(Dense(64, activation = "tanh"))
        model_1.add(Dense(32, activation = "tanh"))
        model_1.add(Dense(16, activation = "tanh"))
        model_1.add(Dense(8, activation = "tanh"))
        model_1.add(Dense(1, activation = "sigmoid"))
        
        model_1.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

        # Define checkpoint callback
        checkpoint_callback = ModelCheckpoint("best_of_weights.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        history_1 = model_1.fit(train_df.values, YTRAIN, epochs=100, validation_data=(valid_df.values, YVALIDATION), callbacks=[checkpoint_callback])
        
        accuracy_after_each_reduction.append({feature: history_1.history['accuracy']})
        Val_accuracy_after_each_reduction.append({feature: history_1.history['val_accuracy']})
        
    return accuracy_after_each_reduction, Val_accuracy_after_each_reduction
        


# In[88]:


accuracy_data, Val_accuracy_data = feature_importance_and_reduction()


# In[144]:


print("Accuracy")
x_labels = []
y_values = []
for feature in accuracy_data:
    feature_name = list(feature.keys())[0]
    accu = feature[feature_name]
    last_accuracy = accu[-1]
    x_labels.append(feature_name)
    y_values.append(last_accuracy)
    print(f"{feature_name}: {last_accuracy}")

plt.figure(figsize=(20, 7))
    
sns.barplot(x=x_labels, y=y_values, palette='hls')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Accuracies after removing each feature')
plt.show()

print()

print("Validation Accuracy")
x_val_labels = []
y_val_values = []
for feature in Val_accuracy_data:
    feature_name = list(feature.keys())[0]
    accu = feature[feature_name]
    last_accuracy = accu[-1]
    x_val_labels.append(feature_name)
    y_val_values.append(last_accuracy)
    print(f"{feature_name}: {last_accuracy}")

plt.figure(figsize=(20, 7))
    
sns.barplot(x=x_val_labels, y=y_val_values, palette='hls')
plt.xticks(rotation=45)
plt.ylabel('Validation Accuracy')
plt.xlabel('Validation Accuracies after removing each feature')
plt.show()
# print("Val Accuracy")
# # last_Val_accuracy = []
# for feature in Val_accuracy_data:
#     feature_name = list(feature.keys())[0]
#     accu = feature[feature_name]
#     last_accuracy = accu[-1]
# #     last_Val_accuracy.append({f"Accuracy Removing {key}": accu})
#     print(f"{feature_name}: {last_accuracy}")
    

