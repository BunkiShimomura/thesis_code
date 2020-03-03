
import os, sys
import glob

# reportlabは解像度の上限があるので，よりクリアな画像が必要な場合は
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A2

from PIL import Image

HELVETICA_TTF = "/Users/Bunki/Desktop/thesis_code/src/fonts/Helvetica_33244fbeea10f093ae87de7a994c3697.ttf"

def get_files(path):
    dir = os.path.dirname(path)
    files = sorted(glob.glob(path + '/*'))

    return files

def set_pdf(files):

    c = canvas.Canvas("arabi_photo.pdf", pagesize = A2, bottomup = False)
    page_width, page_height = A2

    # 文字の規格を設定
    pdfmetrics.registerFont(TTFont('Helvetica', HELVETICA_TTF))
    font_size = 15
    c.setFont('Helvetica', font_size)
    c.setStrokeColor('black')
    c.setFillColor('white')

    x = 0
    y = 0
    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0][0:3]
        img = Image.open(file)
        img = img.resize((int(page_width/8), int(page_width/8)))
        c.drawInlineImage(img, x, y)
        # c.drawImage(file, x, y + int(page_width/8), int(page_width/8), int(page_width/8), mask='auto')
        c.drawCentredString(x+15, y+15, filename)

        x += page_width/8

        if x == page_width:
            x = 0
            y += page_width/8
        else:
            pass
    c.showPage()
    c.save()

    print("done")

if __name__ == '__main__':
    path = input('画像フォルダへの絶対パス: ')
    files = get_files(path)
    pdf = set_pdf(files)
