import argparse

def read_apca_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-subset', default='', type=str)
    parser.add_argument('-cuda', default=0, type=int)
    return parser.parse_args()

def read_jitdp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-project', type=str)
    parser.add_argument('-cuda', default=0, type=int)
    return parser.parse_args()

def read_cmg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str)
    parser.add_argument('-dataset', type=str, default='corec')
    parser.add_argument('-cuda', default=0, type=int)
    return parser.parse_args()