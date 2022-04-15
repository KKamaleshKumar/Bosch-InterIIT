import pandas as pd
import numpy as np
from pandas.core import frame
def new_frame(df, fr_num, id):
    # print(df[df['frame_num'] == fr_num])
    for i, row in df[df['frame_num'] == fr_num].iterrows():
            # print(i)
            df.iloc[i, 1] = id
            id += 1 
    return df
    


def map_bb_id(frame_num, df, last_frame_num = None):
    if last_frame_num == None:
        id = 1
        df = new_frame(df, frame_num, id)       
    else :
        # print(last_frame_num, frame_num-1, last_frame_num != frame_num-1)
        if last_frame_num != frame_num-1:
            id  = max(df[df['frame_num'] == last_frame_num]['person_id'])
            # print(id)
            df = new_frame(df, frame_num, id+1)
        else:
            # print('Inside')
            comparison = df[df['frame_num'] == frame_num-1]
            current = df[df['frame_num'] == frame_num]
            for i, row in current.iterrows():
                min_dist_overall = np.inf
                min_ind = -1
                for j, row_comp in comparison.iterrows():
                    #Algo to get distance and append to distances
                    min_dist = ((df.iloc[i, 2] + df.iloc[i, 5]/2 - (df.iloc[j, 2] + df.iloc[j, 5]/2))**2 + (df.iloc[i, 3] + df.iloc[i, 4]/2- (df.iloc[j, 3]+ df.iloc[i, 4]/2))**2)
                    if min_dist<min_dist_overall:
                        min_dist_overall = min_dist
                        min_ind = j
                if min_dist_overall <400:
                    df.iloc[i, 1] = int(df.iloc[min_ind, 1])
                else:
                    # print(comparison)
                    df.iloc[i, 1] = int(max(comparison['person_id']) + 1)
            #print(frame_num)
    return df

df = pd.read_csv('./video1.csv')
# df = df.iloc[:701, :]
df['person_id'] = np.nan
frames = sorted(list(set(df['frame_num'].values)))
# print(frames, len(frames))
for i in range(len(frames)):
    # if len(df[df['frame_num'] == frame]) == 0:
    #     continue
    if i:
    #    print(frames[i], frames[i-1])
       df = map_bb_id(frames[i], df, frames[i-1])
    else:
        df = map_bb_id(frames[i], df)
    
    # if i == 5:
    #     print(df.head(19))
    #     break
    # print(df)

df.to_csv('./video1fv.csv', index=False)
# print(df.tail())