'''
Author: Aman
Date: 2022-06-21 21:37:41
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-06-24 14:15:32
'''

import multiprocessing
from itertools import product
import datetime
import pickle
# import requests



# def download(file_path, picture_url):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 			(KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE",
#     }
#     r = requests.get(picture_url, headers=headers, timeout=5)
#     with open(file_path, 'wb') as f:
#         f.write(r.content)



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

        ##################################################
        # spilt into multiple chunks if necessary
        size_of_chunk = 0
        out_path = "./"
        if len(inputs) == 1:
            file_list, size_of_chunk = self.split_file(inputs[0], out_path)
        else:
            file_list = inputs

        param = list(product(file_list, [size_of_chunk]))
        ##################################################


        ##################################################
        # if do not split into files, distribute to different process
        size = len(inputs)//self.num_workers
		starts = [i*size for i in range(self.num_workers)]
		param = [(inputs, size, self.num_workers, ) + (start,) for start in starts]
        ##################################################

		res = self.pool.map(self.my_func, param)


        return res


def process_data(process_name, data):
    # define your process data code here
    pass




def MyFunction(param):
    data, size, num_workers, start = params
    print(multiprocessing.current_process().name, 'data processing...')
    
    res = process_data(multiprocessing.current_process().name, data[start:end])

    print(multiprocessing.current_process().name, \
		'data processing done. Success: %d/%d' % (len(res), end - start))
    
    return


def main():
    start_time = datetime.datetime.now()
    print("Start time:", start_time)

    workers_num = 16  # num of multiprocers
    text_file = [""]

    mapper = MyMultiProcess(MyFunction, workers_num)
    res = mapper(inputs=[text_file])

    # aggregate and save
    print("Aggregating and saving...")
    res = [item for sublist in res for item in sublist]
    print("Total res:", len(res))
    save_path = ""
    pickle.dump(res, open(save_path, "wb"))
    print("Aggregating and saving done. File saved to %s." % save_path)

    end_time = datetime.datetime.now()
    print("End time:", end_time)
    print("Total time:", end_time - start_time)


if __name__ == '__main__' :
    main()