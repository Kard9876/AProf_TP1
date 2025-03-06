import csv, sys,re

dataset = "LLM.csv"

#https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text/data

def clearText(texto):
    texto = texto.strip()
    texto = texto[0].upper() + texto[1:]
    return texto if texto.endswith(('.', '?', '!')) else texto + '.'

datasetResult = "dataset_result.csv"

with open(dataset,encoding="utf-8") as f:
    linha = 0
    reader = csv.reader(f,delimiter=",")
    header = next(reader)

    try:
        fd = open(datasetResult, mode='w',newline="", encoding='utf-8')
        writer = csv.writer(fd,delimiter=';')
        writer.writerow(['text', 'label'])

        for row in reader :
            text = row[0]
            label = row[1]
            
            if (label == 'student'):
                num = 0
            elif (label == "ai"):
                num = 1
            else:
                print("Label não identificado")
                print(row)
            
            print(text)
            writer.writerow([f"{clearText(text)}",f"{num}"])

        fd.close()

    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(dataset, reader.line_num, e))