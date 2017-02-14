# -*- coding: utf-8 -*-

from datetime import datetime

def generate_submission(predictions):
    date = datetime.now().strftime('%m-%d-%H:%M:%S')
    fp = open("submissions/%s.csv" % date, "w")
    fp.write("GeneId,Prediction\n")
    for idx, prediction in enumerate(predictions):
        fp.write("%i,%f\n" % ((idx + 1), prediction))
    fp.close()