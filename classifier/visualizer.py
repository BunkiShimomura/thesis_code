# https://qiita.com/yasudadesu/items/1dda5f9d1708b6d4d923
# https://discuss.pytorch.org/t/visualize-live-graph-of-lose-and-accuracy/27267/6
# https://qiita.com/syaorn_13/items/4abdae6fceecccda5d00

import visdom

vis = visdom.Visdom()

def visualizer(loss, acc):
    Acc = 0
    Loss = 0

    Acc += acc
    Loss += loss
