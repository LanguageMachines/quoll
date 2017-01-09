#! /usr/bin/env python

import math
import xlrd
import re
import datetime
from collections import defaultdict

def make_folds(instances,n=10):
    folds = []
    for i in range(n):
        j = i
        fold = []
        while j < len(instances):
            fold.append(instances[j])
            j += n
        folds.append(fold)
    return folds

def make_chunks(lines,nc=16,dist=False):
    i=0
    if dist:
        chunkdict = defaultdict(list)
        while i<(len(lines)):
            if i + nc > len(lines):
                chunkdict[nc-1].extend(lines[i:])
            else:
                for j in range(nc):
                    chunkdict[j].append(lines[i+j])
            i += nc
        chunks = [chunkdict[x] for x in chunkdict.keys()]
    else:
        chunks=[]
        size = int(len(lines)/nc)
        for j in range(nc-1):
            chunks.append(lines[i:(i+size)])
            i += size
        chunks.append(lines[i:])
    return chunks

def cleanup_tweets(tweets, default_length):
    cleaned_tweets = []
    last = default_length - 1
    for i, x in enumerate(tweets):
        fields = x.split('\t')
        if len(fields) == default_length:
            cleaned_tweets.append(x)
        else:
            if len(fields) > default_length:
                for f in fields[default_length:]:
                    fields[last] = fields[last] + ' ' + f
                cleaned_tweets.append('\t'.join(fields))
            else: # len(fields) < 6
                if not i == 0:
                    cleaned_tweets[-1] = cleaned_tweets[-1] + x
    return cleaned_tweets

def format_list(columns,size):
    outlist = []
    f = '{0: <' + size + '}'
    for value in columns:
        outlist.append(f.format(value))
    return outlist

def excel2lines(file_name,sheet_indexes,header = False,annotation=False,date=False):
    lines = []
    num_annotators = False
    workbook = xlrd.open_workbook(file_name)
    #collect sheets
    sheets = []
    print(sheet_indexes)
    for index in sheet_indexes:
        print(index)
        sheets.append(workbook.sheet_by_index(int(index)))
    else:
        sheets = workbook.sheets()
    #for each sheet
    for sheet in sheets:
        sheetlines = []
        if header:
            first_row = 1
        else:
            first_row = 0
        print("gen_functions",sheet.nrows)
        last_row = sheet.nrows
        #iterate the lines
        for rownum in range(first_row,last_row):
            values = []
            #collect annotation values
            if annotation:
                for value in sheet.row_values(rownum):
                    if not type(value) == float:
                        value = value.strip()
                    try:
                        if float(value) in range(2):
                            values.append(float(value))
                    except ValueError:
                        continue
                if num_annotators:
                    if len(values) != num_annotators:
                        print("number of annotation values on line",
                            rownum,"is not consistent; check the inputfile. Exiting...")
                        exit()
                else:
                    num_annotators = len(values)
                    print(num_annotators, "annotators")
            else:
                rowvals = sheet.row_values(rownum)
                if date:
                    try:
                        rowvals[date] = datetime.date(*xlrd.xldate_as_tuple(\
                            sheet.cell_value(rownum,date), workbook.datemode)[:3])
                    except:
                        continue
                values = [unicode(x) for x in rowvals]
            sheetlines.append(values)
        #each sheet is a list of lists
        lines.append(sheetlines)
    return lines

def excel2columns(file_name):
    word_cat = {}
    workbook = xlrd.open_workbook(file_name)
    sheet = workbook.sheet_by_index(0)
    for column in range(sheet.ncols):
        values = sheet.col_values(column)
        header = values[0]
        for i,value in enumerate(values[1:]):
            if value == '':
                break
            word_cat[value] = header
    return word_cat

def read_lcs_files(partsfile,filesdir):
    #generate list of filenames
    parts = open(partsfile)
    filenames = []
    for line in parts.readlines():
        filenames.append(filesdir + line.split(" ")[0])
    parts.close()
    #generate list of wordsequences from file content
    featuresets = []
    for filename in filenames:
        txtfile = open(filename,"r",encoding = "utf-8")
        features = []
        for line in txtfile.readlines():
            features.append(line.strip())
        featuresets.append(features)
        txtfile.close()
    return featuresets
