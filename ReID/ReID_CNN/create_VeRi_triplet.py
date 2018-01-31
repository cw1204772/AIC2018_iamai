import argparse
import numpy as np

if __name__ == '__main__':
    # Argparse
    parser = argparse.ArgumentParser(description='Triplet generator for VeRi dataset')
    parser.add_argument('--db_txt', help='database txt')
    parser.add_argument('--output', help='the output txt file listing all triplets')
    args = parser.parse_args()

    db = np.loadtxt(args.db_txt, dtype=str)
    label = db[:, 1].astype(int)
    out = []
    for i in range(label.shape[0]):
        l = label[i]
        pos = np.nonzero(label == l)
        rng = np.random.permutation(pos[0])
        pos = rng[1] if rng[0] == i else rng[0]
        neg = np.nonzero(label != l)
        rng = np.random.permutation(neg[0])
        neg = rng[0]
        out.append([i, pos, neg])
    np.savetxt(args.output, np.array(out, dtype=int), fmt='%d')
