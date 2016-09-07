#coding:utf-8
import os
import cv2
import numpy as np
def get_date_sum  (file_path) :
    files = os.listdir(file_path)
    return len(files)

# def get_date (file_path):
#     images = []
#     anss = []
#     sum_d = get_date_sum(file_path)
#     for i in range(sum_d):
#         print "{0}/{1}".format(i, sum_d)
#         i, a = get_date_i(file_path, i)
#         images.append(i)
#         anss.append(a)
#     return images, anss

def get_date(file_path, ans_name):
    fp = open(file_path+"/"+ans_name, 'r')
    tmp_line = fp.readline()
    images = []
    anss = []
    refer =np.concatenate([np.arange(26), np.full((6,), 26, dtype=np.int32), np.arange(26)])
    #blank_index = 26
    #print refer
    m = 0
    while tmp_line != "":
        print m
        m+=1
        tmp_lines = tmp_line.split(":")
        tmp_file = tmp_lines[0]
        tmp_ans = tmp_lines[1]
        tmp_ans = tmp_ans[:-1]
        image_shape = (200, 60)
        tmp_path = file_path + "/" + tmp_file
        image = cv2.imread(tmp_path,0)
        image = cv2.resize(image,image_shape,cv2.INTER_CUBIC)
        image = image.transpose((1, 0))
        image = np.resize(image, (200, 60, 1))/255.0
        images.append(image)
        tmp_ans = [refer[ord(tmp_ans[i])-65] if ord(tmp_ans[i])-65 >= 0 and ord(tmp_ans[i])-65 <= 57 else 26 for i in range(len(tmp_ans))]
        anss.append(tmp_ans)
        tmp_line = fp.readline()
    images = np.asarray(images)
    anss = np.asarray(anss)
    return images, anss

def get_ans_maxlen (file_path):
    max_len = 0
    max_i = -1
    max_ans = ""
    files = os.listdir(file_path)
    i = 0
    for f in files :
        ans = f.split('_')[1]
        ans = ans.split('.')[0]
        tmp_len = len(ans)
        if tmp_len > max_len:
            max_len = tmp_len
            max_i = i
            max_ans = ans
        i += 1
    return max_len, max_i, max_ans

def date_difference(y, out):
    out = np.argmax(out, axis=2)
    y = y.tolist()
    out = out.tolist()
    s = len(y)
    right_sum = 0
    for i in range(s):
        y_s = [y[i][m] for m in range(len(y[i]))]
        out_s = [out[i][m] for m in range(len(out[i]))]
        y_i = y_s
        out_i = out_s
        print"{0} \nVS\n {1}\n".format(y_s, out_s)
        if len(y_i)!=len(out_i):
            continue
        else:
            isright = True
            for j in range(len(y_i)):
                if y_i[j]!=out_i[j]:
                    isright = False
                    break
            if isright:
                right_sum+=1
    return right_sum

def SparseDateFrom(x):
    x_ix = []
    x_val = []
    for batch_i, batch in enumerate(x):
        for time, val in enumerate(batch):
            x_ix.append([batch_i, time])
            x_val.append(val)
    x_shape = [len(x), np.asarray(x_ix).max(0)[1] + 1]
    return x_ix, x_val, x_shape

def SparseDatetoDense(x):
    pos = x[0]
    value = x[1]
    index = 0
    print pos
    print value
    res = []
    tmp = []
    for i in range(len(pos)):
        if index == pos[i][0]:
            tmp.append(value[i])
        else:
            index+=1
            res.append(tmp)
            tmp = []
            tmp.append(value[i])
    res.append(tmp)
    return res
