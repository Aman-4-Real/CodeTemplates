'''
Author: Aman
Date: 2022-06-21 21:37:41
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-06-21 22:09:35
'''

import multiprocessing
from itertools import product


class MyMultiProcess(object):
    
    def __init__(self, my_func, num_workers=8):
        self.my_func = my_func
        self.pool = multiprocessing.Pool(num_workers)
        self.num_workers = num_workers
    
    def split_file(self, input_file, out_path):
        """Split a single big file into N chunks where N = num_workers
        """
        file_list = []        
        with open(input_file, 'r') as f_in: # encoding='GB18030', errors='ignore'
            data = f_in.readlines()
            lines_num = len(data)
            size = lines_num // self.num_workers # lines splitted in a chunk
            start = 0
            end = size
            for i in range(lines_num//size):
                chunk_name = "chunk_" + str(i) + ".dat"
                with open(out_path + chunk_name, 'w', encoding='utf-8') as f_out:
                    f_out.write(''.join(data[start:end]))
                start = start + size
                end = end + size
                file_list.append(out_path + "chunk_" + str(i) + ".dat")
        print(f"File splitted into {self.num_workers} chunks.")
        
        return file_list, size

    def __call__(self, inputs):
        if not inputs:
            print("Error! No input files!")
            exit(1)

        # split inputs into chunks
        size_of_chunk = 0
        out_path = "./"
        if len(inputs) == 1:
            file_list, size_of_chunk = self.split_file(inputs[0], out_path)
        else:
            file_list = inputs

        param = list(product(file_list, [size_of_chunk]))
        res = self.pool.map(self.my_func, param)

        return res


def MyFunction(param):
    # define your function here
    pass
    
    return


def main():

    workers_num = 16  # num of multiprocers
    text_file = [""]

    mapper = MyMultiProcess(MyFunction, workers_num)
    res = mapper(inputs=[text_file])

if __name__ == '__main__' :
    main()