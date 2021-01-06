#
# Created on Mon Jan 04 2021
#
# Copyright Unai Garay Maestre, 2021
#

'''
Unit tests for summary API
'''

from fastapi.testclient import TestClient
from summariser.api.server import app

import unittest

class SummariserAPITestCase(unittest.TestCase):
    '''
    Class for running unit tests on the Summariser API
    '''
    @classmethod
    def setUpClass(cls):
        '''
        Set up all tests
        '''
        super(SummariserAPITestCase, cls).setUpClass()
    
    def setUp(self):
        '''
        Set up for each unit test
        '''
        self.client = TestClient(app)

    