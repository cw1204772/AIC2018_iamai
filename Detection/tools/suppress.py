import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_txt_file_path", type=str, default="./Loc1_1_Det_fasta.txt", help="in_txt_file_path")
parser.add_argument("--out_txt_file_path", type=str, default="./selected/Loc1_1_Det_fasta.txt", help="out_txt_file_path")
parser.add_argument("--threshold", type=float, default=1e-5, help="threshold")
parser.add_argument("--upper_area", type=float, default=1e5, help="upper_area")
parser.add_argument("--lower_side", type=float, default=25, help="lower_side")
parser.add_argument("--aspect_ratio", type=float, default=5, help="aspect_ratio")
FLAGS = parser.parse_args()

def get_selected(in_txt_file_path, out_txt_file_path, threshold, upper_area, lower_side, aspect_ratio):
	in_txt_file = open(in_txt_file_path, "rb")
	out_txt_file = open(out_txt_file_path, "wb")
	in_txt_lines = in_txt_file.readlines()
	info_list = list()
	for line in in_txt_lines:
		obj_idx, x, y, w, h, score = line.strip("\n").split(",")[1:7]
		info_list.append((line, float(score), float(w), float(h)))
	for info in info_list:
		if info[1] < threshold or info[2] < lower_side or info[3] < lower_side or info[2] * info[3] > upper_area or info[2] / info[3] > aspect_ratio or info[3] / info[2] > aspect_ratio:
			continue
		out_txt_file.write(info[0])
	in_txt_file.close()
	out_txt_file.close()

if __name__ == "__main__":
	get_selected(FLAGS.in_txt_file_path, FLAGS.out_txt_file_path, FLAGS.threshold, FLAGS.upper_area, FLAGS.lower_side, FLAGS.aspect_ratio)

