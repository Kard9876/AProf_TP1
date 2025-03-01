import csv, sys,re

dataset = "model_training_dataset.csv"

datasetTraining = "dataset_training_small.csv"
datasetValidation = "dataset_validation_small.csv"
datasetTest = "dataset_test_small.csv"

#https://huggingface.co/datasets/dmitva/human_ai_generated_text

def clearText(texto):
    texto = re.sub(";", ".", texto)
    return re.sub("\n", '', texto)

with open(dataset,encoding="utf-8") as f:
    linha = 0
    reader = csv.reader(f,delimiter=",")
    header = next(reader)

    try:
        fd1 = open(datasetTraining, mode='w',newline="", encoding='utf-8',)
        writerTraining = csv.writer(fd1,delimiter=';')
        writerTraining.writerow(['text', 'ai_generator'])

        fd2 = open(datasetValidation, mode='w',newline="", encoding='utf-8')
        writerValidation = csv.writer(fd2,delimiter=';')
        writerValidation.writerow(['text', 'ai_generator'])

        fd3 = open(datasetTest, mode='w',newline="", encoding='utf-8')
        writerTest = csv.writer(fd3,delimiter=';')
        writerTest.writerow(['text', 'ai_generator'])

        for row in reader:
            if linha < 6000:
                writerTraining.writerow([f"{clearText(row[1])}","0"])
                writerTraining.writerow([f"{clearText(row[2])}","1"])

            elif 6000 <= linha < 8000:
                writerTest.writerow([f"{clearText(row[1])}", "0"])
                writerTest.writerow([f"{clearText(row[2])}", "1"])

            elif linha >= 8000:
                break
                writerValidation.writerow([f"{clearText(row[1])}", "0"])
                writerValidation.writerow([f"{clearText(row[2])}", "1"])

            linha += 1

        fd1.close()
        fd2.close()
        fd3.close()

    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(dataset, reader.line_num, e))
