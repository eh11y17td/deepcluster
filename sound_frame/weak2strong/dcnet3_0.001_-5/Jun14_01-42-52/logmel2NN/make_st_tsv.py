import pandas as pd
import os

# def concat_tsv():
weak2strong="./metadata/weak2strong_NN.tsv"
FILE="./metadata/weak2strong_NN_st.tsv"

w2s_df = pd.read_csv(weak2strong, header=0, sep="\t")
w2s_df.insert(1, 'onset', 0)
w2s_df.insert(2, 'offset', 0)
w2s_df = w2s_df.rename(columns={'event_labels': 'event_label'})
# print(w2s_df)

for i, filename in enumerate(w2s_df['filename']):
    file_list = filename.strip('.wav').split('_')
    onset, offset = file_list[-2], file_list[-1]
    # print(filename)
    # print(FILE)
    # print(onset, offset)
    w2s_df.iloc[i, 1] = onset
    w2s_df.iloc[i, 2] = offset
    # exit()
# print(w2s_df)
if os.path.isfile(FILE):
    os.remove(FILE)

# print("sdfafaf")
w2s_df.set_index('filename').to_csv(FILE, sep='\t')