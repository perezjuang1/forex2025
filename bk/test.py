
import pandas as pd

def readData():
    df = pd.DataFrame({'idx': [], 'text': []})
    with open('coding_qual_input.txt', 'r') as file:
        data_array = file.read().splitlines()
    for i in range(len(data_array)):
        row = str(data_array[i]).split(' ') 
        new_row = {'idx': row[0], 'text': row[1]}
        df = df.append(new_row, ignore_index=True)
    return df

def generate_pyramid_and_Decoding(n, df):
    num = 1
    for i in range(1, n + 1):
        for j in range(i):
            #print(num, end=" ")
            num += 1
        print()
        df_row = df.loc[df['idx'] == str(num - 1)]

        idx = df_row['idx']
        text = df_row['text']
        print(f"{idx.values[0]}: {text.values[0]}")

generate_pyramid_and_Decoding(15, readData())