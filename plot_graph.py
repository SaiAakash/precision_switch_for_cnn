import pandas as pd
import matplotlib.pyplot as plt

# Change names of the files accordingly

df1 = pd.read_csv('training_log_f32.csv')
df2 = pd.read_csv('training_log_f64.csv')
df3 = pd.read_csv('training_log_mpt.csv')

acc_f32 = df1.loc[:, 'Accuracy'] 
acc_f64 = df2.loc[:, 'Accuracy'] 
acc_mpt = df3.loc[:, 'Accuracy'] 
batch_index = df1.loc[:, 'Batch']

fig = plt.figure()
plt.plot(batch_index, acc_f32)
plt.plot(batch_index, acc_f64)
plt.plot(batch_index, acc_mpt)
plt.xlabel('Batch Index')
plt.ylabel('Accuracy')
plt.legend(["Float32", "Float64", "MPT"])
plt.show()