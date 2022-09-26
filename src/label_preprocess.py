import glob
import re
import csv
import pickle

def label_preprocess(label_dir: str='ROHAN4600_zundamon_voice_label',
                     dist_dir: str='labels'):
    all_files = glob.glob('train/' + label_dir + "/*")
    label_dict = {}
    label_dict_path = 'train/label.pkl'
    label_count_id = 0
    for filename in all_files:
        with open(filename, 'r') as f:
            labels = f.readlines()
        pattern = re.compile(r'\d+')
        labels = [pattern.sub('', e.replace(' ', '').replace('\n', '')) for e in labels]
        for label in labels:
            if not (label in label_dict):
                label_dict[label] = label_count_id
                label_count_id += 1
        outfile = filename.replace('ROHAN4600_zundamon_voice_label', 'labels').replace('.lab', '.csv')
        with open(outfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(labels)
    with open(label_dict_path, "wb") as f:
        pickle.dump(label_dict, f)



if __name__ == "__main__":
    label_preprocess()
    with open('train/label.pkl', 'rb') as f:
        dict_loaded = pickle.load(f)
    print(dict_loaded)