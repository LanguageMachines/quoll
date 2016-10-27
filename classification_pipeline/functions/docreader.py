
import sys
import xlrd
from openpyxl import load_workbook
import csv
import json
import copy
import re
import string

class Docreader:

    def __init__(self):
        self.lines = []

    def parse_doc(self, doc, delimiter = False, header = False, sheet = False, date = False, time = False):
        form = doc[-4:]
        if form == '.txt':
            self.lines = self.parse_txt(doc, delimiter, header)
        elif form == '.xls': 
            self.lines = self.parse_xls(doc, header, date, time)
        elif form == 'xlsx':
            self.lines = self.parse_xlsx(doc, sheet)
        elif form == 'json': # default twitter parse keys
            parse_keys = [['id'], ['user', 'id'], ['user', 'screen_name'], ['user', 'followers_count'], 
            ['user', 'location'], ['created_at'], ['in_reply_to_screen_name'], ['retweeted_status', 'user', 
            'screen_name'], ['text']]
            self.lines = self.parse_json(doc, parse_keys)
        elif form == '.csv':
            self.lines = self.parse_csv(doc)
        else:
            print('File extension not known, exiting program')
            exit()

    def parse_txt(self, doc, delimiter, header):
        if not delimiter:
            delimiter = '\t'
        with open(doc, 'r', encoding = 'utf-8') as fn:
            lines = [x.strip().split(delimiter) for x in fn.readlines()]
        if header:
            lines = lines[1:]
        return lines

    def parse_xls(self, doc, header, date, time):
        """
        Excel reader
        =====
        Function to read in an excel file

        Parameters
        -----
        doc : str
        	Name of the excel file
        header : bool
        	Indicate if the file contains a header
        date : bool / int
        	If one of the excel fields is in date format, specify the index of the column, give False otherwise
        time : bool / int
        	If one of the excel fields is in time format, specify the index of the column, give False otherwise
        Returns
        -----
        lines : list of lists
            Each list corresponds to the cell values of a row
        """
        workbook = xlrd.open_workbook(doc)
        wbsheet = workbook.sheets()[0]
        rows = []
        begin = 0
        if header:
            begin = 1
        for rownum in range(begin, wbsheet.nrows):
            values = wbsheet.row_values(rownum)
            if date == 0 or date:
               try:
                   datefields = xlrd.xldate_as_tuple(wbsheet.cell_value(rownum, date), workbook.datemode)[:3]
                   values[date] = datetime.date(*datefields)
               except TypeError:
                   values[date] = values[date]           
            if time == 0 or time:
               try:
                   timefields = xlrd.xldate_as_tuple(wbsheet.cell_value(rownum, time), workbook.datemode)[3:]
                   values[time] = datetime.time(*timefields)
               except TypeError:
                   values[time] = values[time]        
            rows.append(values)
        return rows
        
    def parse_xlsx(self, doc, sh):
        workbook = load_workbook(filename = doc)
        if sh:
            sheet = workbook[sh]
        else:
            sheet = workbook['sheet1']
        dimensions = sheet.dimensions
        d1, d2 = dimensions.split(':')
        cols = list(string.ascii_uppercase)
        firstcol = ''.join([x for x in d1 if re.search(r'[A-Z]', x)])
        lastcol = ''.join([x for x in d2 if re.search(r'[A-Z]', x)])
        firstrow = int(''.join([x for x in d1 if re.search(r'[0-9]', x)]))
        lastrow = int(''.join([x for x in d2 if re.search(r'[0-9]', x)]))
        cols = cols[:cols.index(lastcol) + 1]
        lines = []
        for i in range(firstrow, lastrow):
            line = []
            for c in cols:
                line.append(sheet[c + str(i)].value)
            lines.append(line)
        return lines

    def parse_csv(self, doc, delim=','):
        """
        Csv reader
        =====
        Function to read in a csv file
        
        Parameters
        -----
        doc : str
            The name of the csv file

        Returns
        -----
        lines : list of lists
            Each list corresponds to the cell values of a row
        """
        csv.field_size_limit(sys.maxsize)
        try:
            lines = []
            with open(doc, 'r', encoding = 'utf-8') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter = delim)
                for line in csv_reader:
                    lines.append(line)
        except:
            lines = []
            csvfile = open(doc, 'r', encoding = 'utf-8')
            csv_reader = csv.reader(line.replace('\0','') for line in csvfile.readlines())       
            for line in csv_reader:
                lines.append(line)
        return lines

    def parse_json_object(self, obj, keys):
        if keys[0] in obj.keys():
            value = obj[keys.pop(0)]
        else:
            value = ''
            return value
        if len(keys) == 0:
            return value
        else:
            return self.parse_json_object(value, keys)

    def parse_json(self, doc, parse_keys):
        lines = []       
        with open(doc, encoding = 'utf-8') as fn:
            for obj in fn.readlines():
                line = []
                pks = copy.deepcopy(parse_keys)
                decoded = json.loads(obj)
                for keys in pks:
                    line.append(self.parse_json_object(decoded, keys))
                lines.append(line)
        return lines

    def set_lines(self, fields, columndict):
        """
        Columnformatter
        =====
        Function to set columns in the standard format
        
        Parameters
        -----
        columndict : dict
            dictionary to specify the column for the present categories

        Attributes
        -----
        columns : dict
            Dictionary with the standard column for each category
        defaultline : list
            Standard line that is copied for each new line
            Categories that are not present are left as '-'

        Returns
        -----
        new_lines : list of lists
            The correctly formatted lines

        """
        fields = ['label', 'doc_id', 'author_id', 'date', 'time', 'authorname', 'text', 'tagged'] 
        defaultline = ["-"] * len(fields)
        other_header = []
        for key, value in sorted(columndict.items()):
            if not value in fields:
                other_header.append(value)
        if len(other_header) > 0:
            other = True
            other_lines = [other_header]
        else:
            other = False
            other_lines = False

        new_lines = [fields]
        for i, line in enumerate(self.lines):
            new_line = defaultline[:]
            if other:
                other_line = []
            for key, value in sorted(columndict.items()):
                if value in fields:
                    try:
                        new_line[fields.index(value)] = line[key]
                    except:
                        print('field error on line', i, ',',key, value, line, new_line)
                else:
                    other_line.append(line[key])
            new_lines.append(new_line)
            if other:
                other_lines.append(other_line)
        return new_lines, other_lines
