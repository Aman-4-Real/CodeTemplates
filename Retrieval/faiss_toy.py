'''
Author: Aman
Date: 2022-06-27 19:27:24
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-08-25 12:23:20
'''

import os
import logging
logger = logging.getLogger(__name__)

from argparse import ArgumentParser

import numpy as np
import faiss
from faiss import normalize_L2
from tqdm import tqdm
import datetime
import pickle
from PIL import Image


class FaissRetrieval:
    def __init__(self, paths, embs, args, GPU=True):
        self._paths = paths
        self._embs = embs
        self._args = args
        self._GPU = GPU
        self._gpu_id = args.gpu_id
        self.nlist = 3000 # self.nlist is the number of centroids for the quantizer. 4 * sqrt(n) is usually reasonable, or some other O(sqrt(n)).
        self._load_index_path = args.load_index_path
        self._save_index_path = args.save_index_path

        if not self._load_index_path:
            print("Index not given. Constructing a new one!")
            self.construct_index()
            print("Index constructed.")
            print("The number of candidate emb is:", self._index_gpu.ntotal)
        else:
            t1 = datetime.datetime.now()
            print("Loading existing index...")
            self._index = faiss.read_index(self._load_index_path)
            # if self._GPU:
            #     res = faiss.StandardGpuResources()
            #     self._index_gpu = faiss.index_cpu_to_gpu(res, self._gpu_id, self._index)
            print("Index loaded.")
            t2 = datetime.datetime.now()
            print("Time taken:", t2 - t1)
            print("Adding candidates...")
            self.add_candidates(self._paths, self._embs)
            print("The number of candidate emb is:", self._index.ntotal)
        


    def construct_index(self):
        dim = self._embs.shape[-1]
        code_size = 32 # m is the number of subquantizers per codebook
        nbits = 8 # nbits is the number of bits per subquantizer
        quantizer = faiss.IndexFlatL2(dim)
        # index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, code_size, nbits, faiss.METRIC_INNER_PRODUCT)
        # The code_size is typically a power of two between 4 and 64. Like for IndexPQ, d should be a multiple of m.
        if self._GPU:
            res = faiss.StandardGpuResources()
            self._index_gpu = faiss.index_cpu_to_gpu(res, self._gpu_id, index)

        normalize_L2(self._embs)
        
        assert not self._index_gpu.is_trained
        print("Training index...")
        self._index_gpu.train(self._embs)
        print("Index trained.")
        assert self._index_gpu.is_trained

        print("Adding embs...")
        self._index_gpu.add(self._embs)
        print("Embs added.")

        # Save index
        print("Saving index...")
        _index_cpu = faiss.index_gpu_to_cpu(self._index_gpu)
        faiss.write_index(_index_cpu, self._save_index_path)
        print("Index saved.")


    def add_candidates(self, add_path, add_emb):
        dim = add_emb.shape[-1]
        print("Adding embs...")

        normalize_L2(add_emb)
        self._index.add(add_emb)

        # add_index_gpu.add(add_emb)
        print("Embs added.")

        # add_index_cpu = faiss.index_gpu_to_cpu(add_index_gpu)
        # ### Merge the two indices
        # print("Merging the two indices...")
        # self._index.merge_from(add_index_cpu, add_emb.shape[0])
        # print("Merged.")

        # Save index
        print("Saving NEW index...")
        # _index_cpu = faiss.index_gpu_to_cpu(self._index_gpu)
        # faiss.write_index(self._index, self._save_index_path)
        faiss.write_index(self._index, self._save_index_path)
        print("NEW Index saved.")


    def query(self, emb_query, topk=5):             
        self._index_gpu.nprobe = 2048 # centroid to search for neighbors, default nprobe is 1, try a few more
        print("_index_gpu.ntotal:", self._index_gpu.ntotal)
        print("_index_gpu.nprobe:", self._index_gpu.nprobe)
       
        if len(emb_query.shape) == 1:
            emb_query = np.array([emb_query])
        else:
            raise ValueError("emb_queries must be a 1D array")
        normalize_L2(emb_query)
        Dists, Idxes = self._index_gpu.search(emb_query, topk)
        cddts = [[self._paths[idx] for idx in seg] for seg in Idxes]
        return cddts, Dists[0]

    

class Retriever():
    def __init__(self, img_paths, embs, args):
        print("In building INDEX:", len(img_paths), len(embs))
        embs = np.array(embs, dtype = 'float32')
        self.emb_sen_index = FaissRetrieval(img_paths, embs, args, GPU=True)

    def retrieve(self, feat, topk):
        result, distances = self.emb_sen_index.query(feat, topk=topk)
        if len(result) != 0:
            return result[0], distances
        else:
            return None


if __name__ == "__main__":

    # add args
    parser = ArgumentParser()
    parser.add_argument("--load_index_path", type=str, default='', help="Path to load the index file.")
    parser.add_argument("--save_index_path", type=str, default='', help="Path to save the index file.") # INDEXES/misc.index
    parser.add_argument("--keys_save_path", type=str, default='', help="Path to the keys save file.")
    parser.add_argument("--gpu_id", type=int, default=7, help="GPU id.")
    args = parser.parse_args()


    print("Aggregating features...")

    # add your feat paths here
    file_paths = []

    all_imgs_dict = {}
    for file_path in tqdm(file_paths):
        curr_dataset_feats = {}
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if file.endswith('.pkl'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        feats = pickle.load(f)
                        feats = {k: v for k, v in feats.items() if len(v) == 768}
                        curr_dataset_feats.update(feats)
                    print("The number of features in {} is: {}".format(file_path, len(feats)))
        all_imgs_dict.update(curr_dataset_feats)

    print("The number of features in all datasets is:", len(all_imgs_dict))


    print("Constructing INDEX... Using GPU:", args.gpu_id)
    start = datetime.datetime.now()
    retriever = Retriever(list(all_imgs_dict.keys()), list(all_imgs_dict.values()), args)
    print("  Done!")

    print("Saving keys of embs...")
    keys_to_save = list(all_imgs_dict.keys())
    f = open(args.keys_save_path, 'wb')
    pickle.dump(keys_to_save, f)
    f.close()
    print('  Done!')
    
    end = datetime.datetime.now()
    print('Total time is ', end - start)


