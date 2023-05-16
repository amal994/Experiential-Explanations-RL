import os
import sys

if __name__ == "__main__":
    os.system("nohup sh -c '" +
              sys.executable + " train.py --bad_obj purple > res-purple.txt && " +
              sys.executable + " train.py --bad_obj blue > res-blue.txt" +   # Last one shouldn't have &&
              "' &")
