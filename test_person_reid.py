import os
from reid.person_reid import personReIdentifier

if __name__ == '__main__':
    path_dataset='/home/josemiki/FF-PRID-2020/reid/video1'
    reid = personReIdentifier()
    reid.PersonReIdentification(path_dataset,4)