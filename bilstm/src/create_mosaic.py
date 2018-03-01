import os
import json
import numpy as np
import cv2

def create_mosaic(images, border, direction, correct=None, predicted=None):
    """
    direction 0 ==> horizontal
    direction 1 ==> vertical
    """

    big_size = np.sum(np.array([i.shape for i in images]), 0)[:2] + border * 2 * len(images)
    max_size = np.max(np.array([i.shape for i in images]), 0) + border * 2
    if direction == 0:
        mosaic = np.zeros((max_size[0], big_size[1], 3), np.float32)
    if direction == 1:
        mosaic = np.zeros((big_size[0], max_size[1], 3), np.float32)

    start = border

    for i, image in enumerate(images):
        ih, iw, _ = image.shape
        if direction == 0:
            if correct is not None:
                if i == correct:
                    mosaic[0: ih + 2 * border, start - border : start + border + iw, :] =  [0, 255, 0]
            if predicted != correct:
                if i == predicted:
                    mosaic[0: ih + 2 * border, start - border : start + border + iw, :] =  [0, 0, 255]
            mosaic[border : ih + border, start : start + iw, :] = image
            start += border * 2 + iw

        if direction == 1:
            if correct is not None:
                if i == correct:
                    mosaic[start - border : start + border + ih, 0 : iw + 2 * border, :] =  [0, 255, 0]
            if predicted != correct:
                if i == predicted:
                    mosaic[start - border : start + border + ih, 0 : iw + 2 * border, :] =  [0, 0, 255]
            mosaic[start : start + ih, border : iw + border, :] = image
            start += border * 2 + ih

    return mosaic


def create_mosaic_hl(images, border, direction, positions):
    """
    direction 0 ==> horizontal
    direction 1 ==> vertical
    """

    big_size = np.sum(np.array([i.shape for i in images]), 0)[:2] + border * 2 * len(images)
    max_size = np.max(np.array([i.shape for i in images]), 0) + border * 2
    if direction == 0:
        mosaic = np.zeros((max_size[0], big_size[1], 3), np.float32)
    if direction == 1:
        mosaic = np.zeros((big_size[0], max_size[1], 3), np.float32)

    start = border

    for i, image in enumerate(images):
        ih, iw, _ = image.shape
        if direction == 0:
            if i in positions:
                mosaic[0: ih + 2 * border, start - border : start + border + iw, :] =  [0, 255, 0]
            mosaic[border : ih + border, start : start + iw, :] = image
            start += border * 2 + iw
    return mosaic


def create_img_outfit(outfit, positions, savepath, border=5):
    outfit_names = ['data/images/' + i.replace('_', '/') + '.jpg' for i in outfit]
    outfit_prods = [cv2.imread(img) for img in outfit_names]
    outfit = create_mosaic_hl(outfit_prods, border, 0, positions)
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    cv2.imwrite(savepath, outfit.astype(np.uint8))

def create_img_fitb(outfit, predicted, savepath, border=5):
    position = outfit['blank_position'] - 1

    question_names = outfit['question']
    question_names = ['data/images/' + i.replace('_', '/') + '.jpg' for i in question_names]
    questions = [cv2.imread(img) for img in question_names]
    questions_mean_size = np.mean(np.array([i.shape for i in questions]), 0).astype(np.int)
    blank_image = np.zeros(questions_mean_size).astype(np.uint8)
    questions_wblank = [[]] * (len(questions) + 1)
    questions_wblank[:position] = questions[:position]
    questions_wblank[position] = blank_image
    questions_wblank[position + 1 :] = questions[position:]

    answers_names = outfit['answers']
    answers_names = ['data/images/' + i.replace('_', '/') + '.jpg' for i in answers_names]
    answers = [cv2.imread(img) for img in answers_names]

    question_mos = create_mosaic(questions_wblank, border, 0)
    answers_mos = create_mosaic(answers, border, 0, 0, predicted)
    mosaic = create_mosaic([question_mos, answers_mos], 10, 1)
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    cv2.imwrite(savepath, mosaic.astype(np.uint8))


if __name__ == '__main__':

    outs = json.load(open('data/label/fill_in_blank_test.json'))
    basepath = '/tmp/fitbs'
    savepath = os.path.join(basepath, 'out0.jpg')
    create_img(outs[0], 0, savepath)

    # plt.imshow(cv2.cvtColor(tensor.astype(np.uint8), cv2.COLOR_BGR2RGB))
    # plt.show()
