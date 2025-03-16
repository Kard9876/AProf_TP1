import csv, sys,re
import os

pasta = "./datasets"
datasets = os.listdir(pasta)

datasetIA = open("../dataset_IA.csv","w",encoding="utf-8",newline="")

datasetIACSV = csv.writer(datasetIA, delimiter='\t',)
datasetIACSV.writerow(['Text','Label'])

for dataset in datasets:
    file = open(os.path.join(pasta, dataset), "r", encoding="utf-8")
    header = file.readline()
    sniffer = csv.Sniffer()
    delimiterDataset = sniffer.sniff(header).delimiter

    reader = csv.reader(file, delimiter=delimiterDataset)

    for row in reader:
        datasetIACSV.writerow([row[1], f"AI"])


datasetIA.close()
