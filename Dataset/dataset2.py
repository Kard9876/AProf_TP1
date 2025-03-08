import csv, sys,re

dataset = "LLM.csv"

#https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text/data

def clearText(texto):
    texto = texto.strip()
    texto = texto[0].upper() + texto[1:]
    return texto if texto.endswith(('.', '?', '!')) else texto + '.'

datasetResultInput = "dataset_result_input.csv"
datasetResultOutput = "dataset_result_output.csv"

with open(dataset,encoding="utf-8") as f:
    linha = 0
    reader = csv.reader(f,delimiter=",")
    header = next(reader)

    try:
        fd1 = open(datasetResultInput, mode='w',newline="", encoding='utf-8')
        writer1 = csv.writer(fd1,delimiter='\t')
        writer1.writerow(['ID', 'Text'])

        fd2 = open(datasetResultOutput, mode='w', newline="", encoding='utf-8')
        writer2 = csv.writer(fd2, delimiter='\t')
        writer2.writerow(['ID', 'Label'])

        i = 0
        for row in reader :
            i += 1

            text = row[0]
            label = row[1]
            
            if label == "student":
                new_label = "Human"
            elif label == "ai":
                new_label = "AI"
            else:
                print("Label não identificado")
                print(row)

            new_id = f"D2-{i}"
            writer1.writerow([new_id, f"{clearText(text)}"])
            writer2.writerow([new_id, f"{new_label}"])

        fd1.close()
        fd2.close()

    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(dataset, reader.line_num, e))