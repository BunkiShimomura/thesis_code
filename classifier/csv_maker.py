import os
import glob
import csv

def csv_maker():

    path0 = input("Path0: ")
    path1 = input("Path1: ")

    paths = [path0, path1]
    file_list = []
    label_list = []
    n = 0

    for path in paths:
        dir_name = os.path.basename(path)

        files = glob.glob(path + "/*")
        print(files)

        for file in files:
            file_list.append(os.path.basename(file))
            label_list.append(n)
        print(file_list)
        print(label_list)
        n += 1

    with open("data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['label', 'path'])
        for i in range(len(file_list)):
            writer.writerow([str(label_list[i]), file_list[i]])

    f.close()

    return f


if __name__ == "__main__":
    csv_maker()
