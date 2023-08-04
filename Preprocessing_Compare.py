from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_csv = pd.read_csv('./zero_point_one_UOSA_Phase0.csv')

departments = data_csv['Department']

counter_array = {}

for i in departments.unique():
    counter_array[i] = departments.value_counts()[i]

counter_array["Average"] = np.average(list(counter_array.values()))
print(counter_array)

fig = plt.figure(figsize = (20, 10))
plt.rcParams.update({'font.size': 5})
 
# creating the bar plot
plt.barh(list(counter_array.keys()), counter_array.values(), color = 'blue') # plotting the points

plt.xlabel("Number of Courses") # naming the x-axis
plt.ylabel("Department")
plt.title("Number of Courses in Departments")
plt.show()

