# General Library Imports
import copy
import string
import math

from Demo_Classes.people_tracking import PeopleTracking

class OOBx844(PeopleTracking):
    def __init__(self):
        PeopleTracking.__init__(self)

    def updateGraph(self, outputDict):
        PeopleTracking.updateGraph(self, outputDict)