# 指定フォルダ中のファイル名を一括でリネームする
# 親ディレクトリ内の新規フォルダに「共通名_通し番号.拡張子」の新ファイルを作成する

# do 'pip3 install os sys glob shutil' before exuting main file
import os, sys
import glob
import shutil

# path: path to image folder
# folder_name: name of new folder
# ext: extension of files in folder
# com_name: new common name for files. new file names are returned as 'com_name + num + ext'
def Name(path, folder_name):
    # create folder to store renamed files
    dir = os.path.dirname(path)
    os.mkdir(dir + '/' + folder_name)
    folder = dir + '/' + folder_name

    # get files, rename, and store
    files = sorted(glob.glob(path + '/*'))
    print(files)
    for file in files:
        filename = os.path.basename(file)
        print(filename)
        copy = folder + '/' + filename + '_' + folder_name + '.' + 'png'
        shutil.copyfile(file, copy)


if __name__ == '__main__':
    path = input('画像フォルダへの絶対パス: ')
    folder_name = input('共通名: ')
    Name(path, folder_name)
