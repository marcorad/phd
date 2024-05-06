import bibtexparser as bib

with open('references.bib', 'r') as refs:
    lib: bib.bibdatabase.BibDatabase = bib.load(refs)
    
entries = lib.get_entry_list()

title = lambda x: x['title'].lower().replace("{", "").replace("}", "").strip()

entry_consol = {}

# for e in entries:
#     print(e.keys())

#check titles
print('------------')
for entry in entries:
    k = title(entry)
    if k in entry_consol.keys():
        print(f"{entry['ID']} is a duplicate of {entry_consol[k]['ID']}")
    else:
        entry_consol[k] = entry
print('------------')        
#remove abracts and add title brackets
for k, v in entry_consol.items():
    if 'abstract' in v.keys():
        del v["abstract"]
    v['title'] = "{" + v['title'] + "}"
        
import pprint

#sort according to authors
entry_sorted = sorted(entry_consol.values(), key=lambda x: x['author'].lower().replace("{", "").replace("}", "").strip())

# pprint.pprint(entry_sorted)

sorted_lib = bib.bibdatabase.BibDatabase()
sorted_lib.entries = entry_sorted

with open('references_consolodated.bib', 'w') as file:
    bib.dump(sorted_lib, file)
    
