import csv, sys,re
from os import mkdir

try:
    mkdir("./DatasetsGerados")
except FileExistsError:
    pass

datasetTraining_input = "./DatasetsGerados/dataset_training_input.csv"
datasetValidation_input = "./DatasetsGerados/dataset_validation_input.csv"
datasetTest_input = "./DatasetsGerados/dataset_test_input.csv"

datasetTraining_output = "./DatasetsGerados/dataset_training_output.csv"
datasetValidation_output = "./DatasetsGerados/dataset_validation_output.csv"
datasetTest_output = "./DatasetsGerados/dataset_test_output.csv"

sizeDTrain = 3000
sizeDVal = 1000
sizeDTest = 1000

def posicaoDataset(nome_dataset):
    return 0, None, 1
    """
    if nome_dataset == "dataset_IA.csv":
        return 0, None, 1

    elif nome_dataset == "abstracts.textos_final5000.csv":
        return 0, None, 1

    else:
        print("Alguem esqueçeu-se de adicionar o dataset na função posicaoDataset")
        sys.exit()
    """

def is_ai_generator(text):
    return "AI" if str(text).upper == "TRUE" or str(text) == "1" or str(text).upper() == "AI" or str(text).upper() =="YES" else "Human"

def updatedatasets(fd1,fd2,fd3,fd4,fd5,fd6):
    writerTraining_input = csv.writer(fd1, delimiter='\t')
    writerTraining_input.writerow(['ID', 'Text'])

    writerTraining_output = csv.writer(fd2, delimiter='\t')
    writerTraining_output.writerow(['ID', 'Label'])

    writerTest_input = csv.writer(fd3, delimiter='\t')
    writerTest_input.writerow(['ID', 'Text'])

    writerTest_output = csv.writer(fd4, delimiter='\t')
    writerTest_output.writerow(['ID', 'Label'])

    writerValidation_input = csv.writer(fd5, delimiter='\t')
    writerValidation_input.writerow(['ID', 'Text'])

    writerValidation_output = csv.writer(fd6, delimiter='\t')
    writerValidation_output.writerow(['ID', 'Label'])

    return writerTraining_input, writerTraining_output, writerTest_input, writerTest_output, writerValidation_input, writerValidation_output

def createdatasets():
    fd1 = open(datasetTraining_input, mode='w',newline="", encoding='utf-8')

    fd2 = open(datasetTraining_output, mode='w', newline="", encoding='utf-8')

    fd3 = open(datasetTest_input, mode='w', newline="", encoding='utf-8')

    fd4 = open(datasetTest_output, mode='w', newline="", encoding='utf-8')

    fd5 = open(datasetValidation_input, mode='w', newline="", encoding='utf-8')

    fd6 = open(datasetValidation_output, mode='w', newline="", encoding='utf-8')

    return fd1, fd2, fd3, fd4, fd5, fd6

def clearText(texto):
    texto = re.sub(";", ".", texto)
    texto =  re.sub(r'\\cite{([^}]+)}', r'\1', texto)
    texto = re.sub(r'\\text{([^}]+)}', r'\1', texto)
    texto.upper()
    return re.sub("\n", '', texto)


idTest = 0
idTrain = 0
idVal = 0

write = 0

testLimite = False
trainLimite = False
valLimite = False

fd1,fd2,fd3,fd4,fd5,fd6 = createdatasets()
writerTraining_i, writerTraining_o, writerTest_i,writerTest_o, writerValidation_i, writerValidation_o = updatedatasets(fd1,fd2,fd3,fd4,fd5,fd6)

fdReadHuman = open("abstracts.textos_final5000.csv", mode='r', encoding='utf-8')

headerReadHuman = fdReadHuman.readline()
snifferHuman = csv.Sniffer()
delimiterDataset = snifferHuman.sniff(headerReadHuman).delimiter
readerHumanCSV = csv.reader(fdReadHuman, delimiter=delimiterDataset)

fdReadIA = open("dataset_IA.csv", mode='r', encoding='utf-8')
headerReadIA = fdReadIA.readline()
snifferIA = csv.Sniffer()
delimiterDataset = snifferIA.sniff(headerReadIA).delimiter
readerIACSV = csv.reader(fdReadIA, delimiter=delimiterDataset)


pos_texto, pos_texto2, pos_qual = posicaoDataset("dataset_IA.csv")

for rowIA, rowHuman in zip(readerIACSV, readerHumanCSV):
    try:
        if testLimite and trainLimite and valLimite:
            fd1.close()
            fd2.close()
            fd3.close()
            fd4.close()
            fd5.close()
            fd6.close()
            sys.exit(0)

        if write % 3 == 0:
            if idTest < sizeDTest:
                idTest += 1
                writerTest_i.writerow([f"D1-{idTest}", f"{clearText(rowIA[pos_texto])}"])
                writerTest_o.writerow([f"D1-{idTest}",  "AI"])

                idTest += 1
                writerTest_i.writerow([f"D1-{idTest}", f"{clearText(rowHuman[pos_texto])}"])
                writerTest_o.writerow([f"D1-{idTest}", "Human"])
            else:
                testLimite = True
                write += 1

        if write % 3 == 1:
            if idTrain < sizeDTrain:
                idTrain += 1
                writerTraining_i.writerow([f"D1-{idTrain}", f"{clearText(rowIA[pos_texto])}"])
                writerTraining_o.writerow([f"D1-{idTrain}", f"AI"])

                idTrain += 1
                writerTraining_i.writerow([f"D1-{idTrain}", f"{clearText(rowHuman[pos_texto])}"])
                writerTraining_o.writerow([f"D1-{idTrain}", f"Human"])
            else:
                trainLimite = True
                write += 1

        if write % 3 == 2:
            if idVal < sizeDVal:
                idVal += 1
                writerValidation_i.writerow([f"D1-{idVal}", f"{clearText(rowIA[pos_texto])}"])
                writerValidation_o.writerow([f"D1-{idVal}", "AI"])

                idVal += 1
                writerValidation_i.writerow([f"D1-{idVal}", f"{clearText(rowHuman[pos_texto])}"])
                writerValidation_o.writerow([f"D1-{idVal}", "Human"])

            else:
                valLimite = True

        write += 1

    except csv.Error as e:
        sys.exit("ERROR 541: Nao sei")

fd1.close()
fd2.close()
fd3.close()
fd4.close()
fd5.close()
fd6.close()



