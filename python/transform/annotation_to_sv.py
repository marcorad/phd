import xml.etree.ElementTree as et
import os
import pandas as pd
import numpy as np
from typing import Dict

import xml.dom.minidom as md

from tqdm import tqdm

class SVAnnotator:
    def __init__(self, filepath, df: pd.DataFrame, fs, 
                 start_sample_col = 'Start', end_sample_col = 'End',
                 start_freq_col = 'Start Frequency', end_frequency_col = 'End Frequency', 
                 label_col = 'Class', color = '#000000') -> None:
        self.filepath = filepath
        self.df = df
        self.fs = fs
        self.ss_col = start_sample_col
        self.es_col = end_sample_col
        self.sf_col = start_freq_col
        self.ef_col = end_frequency_col
        self.l_col = label_col
        self.color = color
        
    def write(self):
        sv = et.Element("sv")
        data = et.SubElement(sv, "data")
        display = et.SubElement(sv, "display")
        et.SubElement(
            display,
            "layer",
            {
                "id": "6",
                "type": "boxes",
                "name": "Boxes",
                "model": "5",
                "verticalScale": "0",
                "colourName": "",
                "colour": self.color,
                "darkBackground": "true",
            },
        )
        model = et.SubElement( #add start, end, minimum and maximum? 
            data,
            "model",
            {
                "id" : '5',
                'name': '',
                "sampleRate": str(self.fs),
                "type": "sparse",
                "dimensions": "2",
                "resolution": "1",
                "notifyOnAdd": "true",
                "dataset": "4",
                "subtype": "box",
                "units": "Hz",
            },
        )
        dataset = et.SubElement(data, "dataset", {"id": "4", "dimensions": "2"})
        for _, row in self.df.iterrows():
            et.SubElement(
                dataset,
                'point',
                {
                    "frame": str(row[self.ss_col]),
                    "value": str(row[self.sf_col]),
                    "duration": str(row[self.es_col] - row[self.ss_col]),
                    "extent": str(row[self.ef_col] - row[self.sf_col]),
                    "label": str(row[self.l_col]) if self.l_col != None and self.l_col != '' else ''
                },
            )
            
        xmlstr = et.tostring(sv, xml_declaration=False)
        xmlstr = '<?xml version="1.0" encoding="UTF-8"?>  <!DOCTYPE sonic-visualiser>' + str(xmlstr, encoding='ascii')       
        xmlstr = md.parseString(xmlstr).toprettyxml()     
        with open(self.filepath, 'w') as file:
            file.write(xmlstr)



class SVConverter:
    def __init__(self, inputdir, outputdir) -> None:
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.input_annotations = []
        for entry in os.scandir(inputdir):
            if entry.name.endswith(".selections.txt"):
                f = {
                    "file": entry.path,
                    "class": entry.name.split(".selections.txt")[0],
                    "df": pd.read_csv(entry.path, header="infer", sep="\t"),
                }
                if not f["df"].empty:
                    self.input_annotations.append(f)

    def convert(self, fs=250, orig_fs = 1000):
        for a in self.input_annotations:
            a["df"]["Class"] = a["class"]
        all = pd.concat([a["df"] for a in self.input_annotations], axis="index")
        df = pd.DataFrame()
        df["Class"] = all["Class"]
        df["File"] = all["Begin File"]
        df["Start Seconds"] = all["Beg File Samp (samples)"]/orig_fs
        df["End Seconds"] = all["End File Samp (samples)"]/orig_fs
        df["Start"] = (df["Start Seconds"] * fs).astype(np.int32)
        df["End"] = (df["End Seconds"] * fs).astype(np.int32)
        df["Delta Seconds"] = df["End Seconds"] - df["Start Seconds"]
        df["Start Frequency"] = all["Low Freq (Hz)"]
        df["End Frequency"] = all["High Freq (Hz)"]
        self.all_annotations = df
        self.per_file_annotation: Dict[str, pd.DataFrame] = {}

        files = df["File"].unique()
        for f in files:
            self.per_file_annotation[f] = df[df.File == f]

        for fname, df in tqdm(self.per_file_annotation.items()):
            df: pd.DataFrame
            sva = SVAnnotator(self.outputdir + '\\' + fname.split('.wav')[0] + '.svl', df, fs)
            sva.write()
            


if __name__ == "__main__":
    c = SVConverter(
        "D:\Whale Data\AcousticTrends_BlueFinLibrary\casey2017",
        "D:\Whale Data\Datasets\Casey2017\\sonicvis",
    )
    # print(pp.files, pp.orig_fs)
    c.convert()
