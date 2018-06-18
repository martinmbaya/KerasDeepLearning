import sys
import csv

name="winequality-red.csv"
data=open(name, "r")
reader= csv.reader(data)
xList = []
labels = []
names = []
firstLine = True
for line in data:
	if firstLine:
		names = line.strip().split(";")
		print(names)
		firstLine = False
	else:
		#split on semi-colon
		row = line.strip().split(";")
		#put labels in separate array
		labels.append(float(row[-1]))
		#remove label from row
		row.pop()
		#convert row to floats
		floatRow = [float(num) for num in row]
		xList.append(floatRow)

nrow = len(xList)
ncol = len(xList[1])
print("Number of rows is : ", nrow)
print("Number of columns is : ", ncol)