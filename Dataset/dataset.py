import csv, sys,re
from os import mkdir

#LINKS DATASETS
"""
model_training_dataset.csv = https://huggingface.co/datasets/dmitva/human_ai_generated_text
data_set.csv = https://www.kaggle.com/datasets/heleneeriksen/gpt-vs-human-a-corpus-of-research-abstracts/data

"""

"Se adicionarem dataset na linha adicionar fun posicaoDataset (adicionar no inicio)"
listaDatasets = ["output.csv"]

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

sizeDTrain = 10000
sizeDVal = 1000
sizeDTest = 1000

def posicaoDataset(nome_dataset):
    """
    Caso o dataset por casa linha possua 2 textos 1 ia outro humano indicar posiçao dos 2 (primeiro humano e depois ia),
    e o ultimo parametro é retornado a None

    Exemplo:

    id,human_text,ai_text,instructions
    return 1 , 2 , None

    Caso o dataset por casa linha possua 1 texto indicar a posição dele o segundo parametro vai a None e o ultimo indica
    a label onde indentifica se o texto foi gerado por ia ou por um humano.

    Exemplo:

    title,abstract,capybara,is_ai_generated
    return 1,None,3
    """


    if nome_dataset == "data_set.csv":
        return 1, None, 3

    elif nome_dataset == "model_training_dataset.csv":
        return 1, 2, None

    elif nome_dataset == "output.csv":
        return 1 ,2, None

    else:
        print("Alguem esqueçeu-se de adicionar o dataset na função posicaoDataset")
        sys.exit()

#fixme verificar todos os casos
def is_ai_generator(text):
    return "AI" if str(text).upper == "TRUE" or str(text) == "1" or str(text).upper() == "AI" or str(text).upper() =="YES" else "HUMAN"

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

for dataset in listaDatasets:
    with open(dataset, mode="r", encoding="utf-8") as f:
        header = f.readline()
        sniffer = csv.Sniffer()
        delimiterDataset = sniffer.sniff(header).delimiter

        reader = csv.reader(f,delimiter=delimiterDataset)

        pos_texto, pos_texto2, pos_qual = posicaoDataset(dataset)

        try:
            for row in reader:
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
                        if pos_texto2 is not None:
                            idTest += 1
                            writerTest_i.writerow([f"D1-{idTest}", f"{clearText(row[pos_texto])}"])
                            writerTest_o.writerow([f"D1-{idTest}", "HUMAN"])

                            idTest += 1
                            writerTest_i.writerow([f"D1-{idTest}", f"{clearText(row[pos_texto2])}"])
                            writerTest_o.writerow([f"D1-{idTest}", "AI"])

                        else:
                            idTest += 1
                            writerTest_i.writerow([f"D1-{idTest}", f"{clearText(row[pos_texto])}"])
                            writerTest_o.writerow([f"D1-{idTest}", f"{is_ai_generator(row[pos_qual])}"])

                    else:
                        testLimite = True
                        write += 1

                if write % 3 == 1:
                    if idTrain < sizeDTrain:
                        if pos_texto2 is not None:
                            idTrain += 1
                            writerTraining_i.writerow([f"D1-{idTrain}", f"{clearText(row[pos_texto])}"])
                            writerTraining_o.writerow([f"D1-{idTrain}", "HUMAN"])

                            idTrain += 1
                            writerTraining_i.writerow([f"D1-{idTrain}", f"{clearText(row[pos_texto2])}"])
                            writerTraining_o.writerow([f"D1-{idTrain}", "AI"])

                        else:
                            idTrain += 1
                            writerTraining_i.writerow([f"D1-{idTrain}", f"{clearText(row[pos_texto])}"])
                            writerTraining_o.writerow([f"D1-{idTrain}", f"{is_ai_generator(row[pos_qual])}"])
                    else:
                        trainLimite = True
                        write += 1

                if write % 3  == 2:
                    if idVal < sizeDVal:
                        if pos_texto2 is not None:
                            idVal += 1
                            writerValidation_i.writerow([f"D1-{idVal}", f"{clearText(row[pos_texto])}"])
                            writerValidation_o.writerow([f"D1-{idVal}", "HUMAN"])

                            idVal += 1
                            writerValidation_i.writerow([f"D1-{idVal}", f"{clearText(row[pos_texto2])}"])
                            writerValidation_o.writerow([f"D1-{idVal}", "AI"])

                        else:
                            idVal += 1

                            writerValidation_i.writerow([f"D1-{idVal}", f"{clearText(row[pos_texto])}"])
                            writerValidation_o.writerow([f"D1-{idVal}", f"{is_ai_generator(row[pos_qual])}"])

                    else:
                        valLimite = True

                write+=1

        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(dataset, reader.line_num, e))

    f.close()

fd1.close()
fd2.close()
fd3.close()
fd4.close()
fd5.close()
fd6.close()



