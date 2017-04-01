#!/usr/bin/env python

import sys
import re
import os
import random

test =  {"\"bbox_pred\"":['num_output: ',4], "\"cls_score\"":["num_output: ",1]}
train = {"\"bbox_pred\"":['num_output: ',4], "input-data":['param_str: \"\'num_classes\': ',1], "roi-data":['param_str: \"\'num_classes\': ', 1], "\"cls_score\"":["num_output: ",1]}

if __name__ == '__main__':
  total_class = int(sys.argv[1]) + 1   # +1 for background
  infile = sys.argv[2]
  if infile.rfind("train.prototxt") != -1:
    replace_dict = train
  else :
    replace_dict = test

  bak_file = "{0}.{1}".format("prototxt",str(random.randint(1,10000)))
  os.popen('cp {0} /tmp/{1}'.format(infile,bak_file))
  sys.stderr.write("bak file to /tmp/{0}\n".format(bak_file))
  f = open(infile).readlines()
  line_num = 0
  found = False
  while line_num < len(f):
    for key in replace_dict.iterkeys():
      if f[line_num].find(key) != -1:
        tmp_num = 1
        found = False
        while tmp_num <= 20 and line_num + tmp_num < len(f):
          index = tmp_num + line_num
          r_str = replace_dict[key][0]
          r_num = replace_dict[key][1] * total_class
          if f[index].find(r_str) != -1:
            f[index] = re.sub(r_str + "[0-9]{1,4}", r_str+ str(r_num), f[index])
            sys.stderr.write("replace {0} in {1} with {2}\n".format(replace_dict[key][0], key, r_num))
            tmp_num +=1 
            found = True
            break
          else:
            tmp_num += 1
        line_num += tmp_num
        if not found: sys.stderr.write("not found {0} {1}\n".format(replace_dict[key][0], key))
        break
      else:
        pass
    line_num += 1

  f_w = open(infile,'w')
  for i in f:
    f_w.write(i)
