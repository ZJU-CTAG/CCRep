# -*- coding:utf-8 -*-
import io
import re


class Counter:
    def __init__(self, re_path, ptg_path, srg_path, ref_path):

        self.mapping = dict() # train frequency
        self.voc = []
        self.correct = []
        # with io.open(train_path, encoding="utf-8") as train:
        #     data = train.read()
        #     words = [s.lower() for s in re.findall("\w+", data)]
        #     for word in words:
        #      self.mapping[word] = self.mapping.get(word, 0) + 1

        with io.open(re_path, encoding="utf-8") as ret:
            self.ret_data = ret.readlines()

        with io.open(ref_path, encoding="utf-8") as ref:
            self.ref_data = ref.readlines()

        with io.open(ptg_path, encoding="utf-8") as ptg:
            self.ptg_data = ptg.readlines()

        with io.open(srg_path, encoding="utf-8") as srg:
            self.srg_data = srg.readlines()

    def get_better(self):
        tmp = []
        for ret, ref, ptg, srg in zip(self.ret_data, self.ref_data, self.ptg_data, self.srg_data):
            ret_l = [s.lower() for s in re.findall("\w+", ret)]
            ref_l = [s.lower() for s in re.findall("\w+", ref)]
            ptg_l = [s.lower() for s in re.findall("\w+", ptg)]
            srg_l = [s.lower() for s in re.findall("\w+", srg)]
            for item in srg_l:
                if item in ref_l and item in ret_l and item not in ptg_l:
                    if ref == srg:
                        continue
                    else:
                        print(srg)

    def get_voc(self):
        tmp = []
        for test in self.test_data:
            test_l = [s.lower() for s in re.findall("\w+", test)]
            for item in test_l:
                tmp.append(item)

        self.voc = list(set(tmp))

    def get_correct(self):
        tmp = []
        for test, ref in zip(self.test_data, self.ref_data):
            test_l = [s.lower() for s in re.findall("\w+", test)]
            ref_l = [s.lower() for s in re.findall("\w+", ref)]
            for item in test_l:
                if item in ref_l:
                    tmp.append(item)

        self.correct = list(set(tmp))
        print(len(self.correct))


    def print_map(self):
        # mapping = sorted(self.mapping.items(), key=lambda item: item[1], reverse=True)
        result = [0,0,0,0,0,0,0,0] # 1 2 3 4 5 10  >10 <=30 >20
        # for tgt, ref in zip(self.tgt_data, self.ref_data):
        #     print(tgt.strip().lower().split())
        for item in self.voc:
            number = self.mapping[item]
            if number == 1:
                result[0] += 1
            elif number == 2:
                result[1] += 1
            elif number <= 5:
                result[2] += 1
            elif number <= 10:
                result[3] += 1
            elif number <= 20:
                result[4] += 1
            elif number <= 50:
                result[5] += 1
            else:
                result[6] += 1

        print(result)


if __name__ == '__main__':

    re_path = "../result/10000retrieval"
    ptg_path = "../result/ptg/cleaned_test.msg"
    srg_path = "../result/SRGen/cleaned_test.msg"
    ref_path = "../result/ref/cleaned_test.msg"
    counter = Counter(re_path, ptg_path, srg_path, ref_path)
    counter.get_better()
    #counter.get_voc()
    #counter.get_correct()
    #counter.print_map()
